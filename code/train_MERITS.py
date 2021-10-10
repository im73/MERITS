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
import json

from models.models import MERITS
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params, CreateDataLoader
from sklearn.model_selection import KFold
from data.DataLoader import DataLoader
from models.ODE import RNNEncoder,ODEVAE


torch.manual_seed(1203)
np.random.seed(1203)

model_name = 'MERITS'
# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--params_path', type=str, help="the path to model parameter")

args = parser.parse_args()

def eval(model, data_test, ddi_adj, eval_diff = True):
    # evaluate
    print('')
    model.eval()
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    case_study = defaultdict(dict)

    test_loader = CreateDataLoader(data_test, batch_size = 32)
    for step, seq_input in enumerate(test_loader):

        y_pred = []
        y_pred_prob = []
        y_pred_label = []

        lab, glu, time, med, length = [], [], [], [], []
        y_gt = []
        for k in range(len(seq_input)):
            if eval_diff and (seq_input[k][3][-1] == seq_input[k][3][-2]):
                continue
            lab.append(seq_input[k][0])
            glu.append(seq_input[k][1])
            time.append(seq_input[k][2])
            med.append(seq_input[k][3])
            length.append(seq_input[k][4])
            y_gt.append(seq_input[k][3][-1])
        # skip the zero batch
        if len(lab) == 0:
            continue
        seq_input = [lab, glu, time, med, length]
        y_gt = np.array(y_gt)

        target_output1 = model(seq_input)

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
    print("DDI Rate: {}, Jaccard: {},  PRAUC: {}, AVG_PRC: {}, AVG_RECALL: {}, AVG_F1: {}".format(ddi_rate,np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)))

    # print('avg med', med_cnt / visit_cnt)

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


def main():
    # load the parameter
    hf = open(os.path.join(args.params_path), 'r')
    params = json.load(hf)

    TARGET_DDI = params['TARGET_DDI']
    T = params['T']
    decay_weight = params['decay_weight']
    batch_size = params['batch_size']
    glu_dim = params['glu_dim']
    emb_dim = params['hidden_size']
    LR = params['learning_rate']
    nb_epoch = params['nb_epoch']
    ddi = params['ddi']
    ODE_path = params['ODE_path']
    eval_diff = params['only_eval_diff']

    Total_epoch = nb_epoch
    Neg_Loss = DDI_IN_MEM = ddi

    model_name = args.model_name + '_learning_rate_' + str(LR) + '_hidden_size_' + str(emb_dim) + '_epoch_' + str(nb_epoch) + '_batch_size_' + str(batch_size)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load the three graph
    ehr_adj_path = './data/ocur.pkl'
    ddi_adj_path = './data/DDI_.pkl'
    time_adj_path = './data/time.pkl'

    ehr_adj = np.array(dill.load(open(ehr_adj_path, 'rb')))
    ddi_adj = np.array(dill.load(open(ddi_adj_path, 'rb')))
    time_adj = np.array(dill.load(open(time_adj_path, 'rb')))
    ehr_adj = np.array(ehr_adj) / np.max(np.array(ehr_adj))
    ddi_adj = np.array(ddi_adj) / np.max(np.array(ddi_adj))
    time_adj = np.array(time_adj) / np.max(np.array(time_adj))

    # prepare data
    data = np.array(DataLoader())
    kf = KFold(n_splits=5, shuffle=True, random_state=2021)


    # load the pre-trained NeuralODE
    # output_size, hidden_size, laten_size
    vae = ODEVAE(1, glu_dim, 16)
    vae.load_state_dict(torch.load(ODE_path))
    for param in vae.parameters():
        param.requires_grad = False
    
    # build the model
    model = MERITS(lab_dim=len(data[1][0]), med_dim=len(data[0][3][0]), ehr_adj=ehr_adj, ddi_adj=ddi_adj,
                    glu_encoder=vae, time_adj = time_adj, emb_dim= emb_dim, device=device, ddi_in_memory=DDI_IN_MEM, glu_dim = glu_dim)
    model.to(device=device)

    print('parameters', get_n_params(model))
    print('hidden_size : {}, learning_rate : {}'.format(emb_dim, LR))

    # create the save dir and save the init model
    if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))
    torch.save(model.state_dict(), open(os.path.join('saved', model_name,'init.model' ), 'wb+'))

    for train_index, test_index in kf.split(data):
        # model init
        model.load_state_dict(torch.load(open('saved/' + model_name +'/init.model', 'rb')))
        optimizer = Adam(list(model.parameters()), lr=LR)
        # get batch data
        data_train = data[train_index]
        data_test = data[test_index]
        train_loader = CreateDataLoader(data_train, batch_size)
        
        history = defaultdict(list)
        best_step = 0
        best_ja = 0
        start_time = tm.time()
        prediction_loss_cnt = 0
        neg_loss_cnt = 0
        step = 0
        epoch = 0
        while epoch < Total_epoch :
            loss_record1 = []
            model.train()
            # Get batch data
            # seq_input = data_train[index:min(index + batch_size, len(data_train))]
            for step, seq_input in enumerate(train_loader):
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
            

            # trained atfer one epoch, start to eval
            
            
            T *= decay_weight
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_test, ddi_adj, eval_diff = eval_diff)

            history['ja'].append(ja)
            history['ddi_rate'].append(ddi_rate)
            history['avg_p'].append(avg_p)
            history['avg_r'].append(avg_r)
            history['avg_f1'].append(avg_f1)
            history['prauc'].append(prauc)

            end_time = tm.time()
            elapsed_time = (end_time - start_time) / 60
            epoch += 1
            print('\tEpoch: %d, Loss: %.4f, One Epoch Time: %.2fm\n' % (epoch, np.mean(loss_record1), elapsed_time,))
            start_time = tm.time()
            if epoch != 0 and best_ja < ja:
                best_epoch = epoch
                best_ja = ja
                torch.save(model.state_dict(), open(
                    os.path.join('saved', model_name, 'Epoch_%d_JA_%.4f_DDI_%.4f.model' % (epoch, ja, ddi_rate)), 'wb+'))

        print('best_epoch:', best_epoch)

if __name__ == '__main__':
    main()
