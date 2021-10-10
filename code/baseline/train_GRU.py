import torch
import argparse
import numpy as np
import dill
import time as tm
import random
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F
from collections import defaultdict

from models import Retain
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params
from sklearn.model_selection import KFold
from DataLoader import DataLoader
from ODE import RNNEncoder

torch.manual_seed(1203)
np.random.seed(1203)

model_name = 'Retain'
resume_name = ''

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--eval', action='store_true', default=False, help="eval mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_name, help='resume path')
parser.add_argument('--ddi', action='store_true', default=False, help="using ddi")

args = parser.parse_args()
model_name = args.model_name
resume_name = args.resume_path


def eval(model, data_eval, ddi_adj, epoch):
    # evaluate
    print('')
    model.eval()
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    case_study = defaultdict(dict)

    batch_size = 32
    index = 0
    for i in range(int(len(data_eval) / batch_size)):

        y_pred = []
        y_pred_prob = []
        y_pred_label = []

        seq_input = data_eval[index:min(index + batch_size, len(data_eval))]
        index = (index + batch_size) % len(data_eval)

        lab, glu, time, med, length = [], [], [], [], []
        y_gt = []
        for k in range(len(seq_input)):
            lab.append(seq_input[k][0])
            glu.append(seq_input[k][1])
            time.append(seq_input[k][2])
            med.append(seq_input[k][3])
            length.append(seq_input[k][4])

            y_gt.append(seq_input[k][3][-1])
        seq_input = [lab, glu, time, med, length]
        y_gt = np.array(y_gt)

        target_output1, _ = model(seq_input)


        target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()
        y_pred_prob = target_output1
        y_pred_tmp = target_output1.copy()
        y_pred_tmp[y_pred_tmp >= 0.5] = 1
        y_pred_tmp[y_pred_tmp < 0.5] = 0
        y_pred = y_pred_tmp

        smm_record += list(y_pred)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred),
                                                                                 np.array(y_pred_prob))
        case_study[adm_ja] = {'ja': adm_ja, 'patient': seq_input, 'y_label': y_pred_label}

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)

    # ddi rate
    ddi_rate = ddi_rate_score(np.array(smm_record), ddi_adj)

    llprint('\tDDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
    ))

    # print('avg med', med_cnt / visit_cnt)

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


def main():
    if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))

    ehr_adj_path = '../data/ocur.pkl'
    ddi_adj_path = '../data/DDI_.pkl'
    device = torch.device('cpu:0')

    ehr_adj = np.array(dill.load(open(ehr_adj_path, 'rb')))
    ddi_adj = np.array(dill.load(open(ddi_adj_path, 'rb')))
    ehr_adj = np.array(ehr_adj) / np.max(np.array(ehr_adj))
    ddi_adj = np.array(ddi_adj) / np.max(np.array(ddi_adj))


    data = np.array(DataLoader())
    kf = KFold(n_splits=5, shuffle=True, random_state=2021)

    EPOCH = 4000
    LR = 0.001
    TEST = args.eval
    Neg_Loss = args.ddi
    DDI_IN_MEM = args.ddi
    TARGET_DDI = 0.05
    T = 0.9
    decay_weight = 0.85
    batch_size = 32

    glu_encoder = RNNEncoder(1, 32, 32)
    model = Retain(med_dim = 12,lab_dim=len(data[1][0]), glu_encoder = glu_encoder,emb_size=64)
    model.to(device=device)

    print('parameters', get_n_params(model))

    torch.save(model.state_dict(), open(
        os.path.join('saved', model_name,
                     'init.model' ), 'wb+'))

    if TEST:

        for train_index, test_index in kf.split(data):
            model.load_state_dict(torch.load(open(resume_name, 'rb')))
            model.to(device=device)
            data_train = data[train_index]
            data_eval = data[test_index]
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_eval, 12, 0)
            llprint('\tDDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
                ddi_rate, np.mean(ja), np.mean(prauc), avg_p, avg_r, np.mean(avg_f1)))
    else:
        for train_index, test_index in kf.split(data):
            # model init
            model.load_state_dict(torch.load(open(os.path.join('saved', model_name,
                                                               'init.model'), 'rb')))
            optimizer = Adam(list(model.parameters()), lr=LR)
            # get batch data
            data_train = data[train_index]
            data_eval = data[test_index]
            random.shuffle(data_train)

            history = defaultdict(list)
            best_epoch = 0
            best_ja = 0
            index = 0
            start_time = tm.time()
            prediction_loss_cnt = 0
            neg_loss_cnt = 0
            for epoch in range(EPOCH):
                loss_record1 = []
                model.train()

                seq_input = data_train[index:min(index + batch_size, len(data_train))]
                index = (index + batch_size) % len(data_train)

                lab, glu, time, med, length = [], [], [], [], []
                loss1_target = []
                for k in range(len(seq_input)):
                    lab.append(seq_input[k][0])
                    glu.append(seq_input[k][1])
                    time.append(seq_input[k][2])
                    med.append(seq_input[k][3])
                    length.append(seq_input[k][4])
                    loss1_target.append(seq_input[k][3][-1])

                loss3_target = np.full((len(loss1_target), len(loss1_target[0])), -1)
                for j in range(len(loss1_target)):
                    idx = 0
                    for k in range(len(loss1_target[j])):
                        if loss1_target[j][k] == 1:
                            loss3_target[j][idx] = k
                            idx += 1

                seq_input = [lab, glu, time, med, length]

                target_output1, batch_neg_loss = model(seq_input)

                loss1 = F.binary_cross_entropy_with_logits(target_output1,
                                                           torch.FloatTensor(loss1_target).to(device))
                loss3 = F.multilabel_margin_loss(F.sigmoid(target_output1),
                                                 torch.LongTensor(loss3_target).to(device))
                if Neg_Loss:
                    target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()

                    target_output1[target_output1 >= 0.5] = 1
                    target_output1[target_output1 < 0.5] = 0

                    current_ddi_rate = ddi_rate_score(target_output1, ddi_adj)
                    if current_ddi_rate <= TARGET_DDI:
                        loss = 0.9 * loss1 + 0.01 * loss3
                        prediction_loss_cnt += 1
                    else:
                        rnd = np.exp((TARGET_DDI - current_ddi_rate) / T)
                        if np.random.rand(1) < rnd:
                            loss = batch_neg_loss
                            neg_loss_cnt += 1
                        else:
                            loss = 0.9 * loss1 + 0.01 * loss3
                            prediction_loss_cnt += 1
                else:
                    loss = 0.9 * loss1 + 0.01 * loss3

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                loss_record1.append(loss.item())


                # annealing
                T *= decay_weight
                if epoch % 50 == 0 and epoch != 0 and epoch > 1000:
                    ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_eval, ddi_adj, epoch)

                    history['ja'].append(ja)
                    history['ddi_rate'].append(ddi_rate)
                    history['avg_p'].append(avg_p)
                    history['avg_r'].append(avg_r)
                    history['avg_f1'].append(avg_f1)
                    history['prauc'].append(prauc)

                    end_time = tm.time()
                    elapsed_time = (end_time - start_time) / 60
                    llprint('\tEpoch: %d, Loss: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                                          np.mean(
                                                                                                              loss_record1),
                                                                                                          elapsed_time,
                                                                                                          elapsed_time * (
                                                                                                                  EPOCH - epoch - 1) / 60 / 16))
                    start_time = tm.time()
                    if epoch != 0 and best_ja < ja:
                        best_epoch = epoch
                        best_ja = ja
                        torch.save(model.state_dict(), open(
                            os.path.join('saved', model_name,
                                         'Epoch_%d_JA_%.4f_DDI_%.4f.model' % (epoch, ja, ddi_rate)), 'wb+'))

            print('best_epoch:', best_epoch)


if __name__ == '__main__':
    main()
