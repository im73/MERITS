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
import sys
import pickle

from models.models import Retain
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params, CreateDataLoader
from models.ODE import RNNEncoder,ODEVAE


torch.manual_seed(1203)
np.random.seed(1203)

model_name = 'Retain'
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
    print("DDI Rate: {}, Jaccard: {},  PRAUC: {}, AVG_PRC: {},AVG_RECALL: {}, AVG_F1: {}"
        .format(ddi_rate,np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)))

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
    model_path = params['Retain_model_path']
    random_seed = params['random_seed']
    sample_data_path = params['sample_data_path']

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
    #prepare data
    data_test = pickle.load(open(sample_data_path, 'rb'))

    # output_size, hidden_size, laten_size
    glu_encoder = RNNEncoder(1, 16, 16)
    model = Retain(med_dim = 12,glu_encoder = glu_encoder, emb_size=64, device = device)
    model.load_state_dict(torch.load(open(model_path, 'rb')))
    model.to(device=device)
    
    # eval
    print("model_name :", model_name)
    ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_test, ddi_adj, eval_diff = eval_diff)
    


if __name__ == '__main__':
    main()
