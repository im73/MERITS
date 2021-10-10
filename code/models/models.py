import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.layers import GraphConvolution, GraphAttentionLayer, Attn_encoder, MultiHeadAttention
from models.tlstm import TimeLSTM
'''
Our model
'''


def get_mask(p_length, emb_dim = 64):
    mask_zeors = np.zeros((len(p_length), 25, emb_dim))
    ones = np.ones((1, emb_dim))
    for i in range(len(p_length)):
        mask_zeors[i, p_length[i] - 1, :] = ones
    return mask_zeors


class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class GAMENet(nn.Module):
    def __init__(self, lab_dim, med_dim, ehr_adj, ddi_adj, glu_encoder, emb_dim=64, device=torch.device('cpu:0'),
                 ddi_in_memory=True):
        super(GAMENet, self).__init__()

        self.lab_dim = lab_dim
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.ddi_in_memory = ddi_in_memory
        self.emb_dim = emb_dim
        # static featrue
        self.lab_embeddings = torch.nn.Parameter(torch.Tensor(np.random.randn(lab_dim, emb_dim)))

        self.static_ll = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(p=0.4)

        # glu_encode
        self.glu_encoder = glu_encoder

        # dynamic_featrue
        self.encoders = nn.GRU(emb_dim, emb_dim, batch_first=True)

        self.query = nn.Sequential(

            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )

        #
        self.ehr_gcn = GCN(voc_size=med_dim, emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=med_dim, emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim * 3),
            nn.ReLU(),
            nn.Linear(emb_dim * 3, med_dim)
        )

    def forward(self, input):
        # get the input
        lab, glu, time, med, length = input
        lab  = torch.Tensor(lab).to(self.device)
        '''I:generate current input'''
        # static featrue
        lab = self.dropout(lab.mm(self.lab_embeddings))  # (1,1,dim)

        static_fea = self.static_ll(lab)
        # dynamic featrue
        glu_seq = torch.Tensor([]).to(self.device)
        history_keys = []
        for i in range(len(glu)):
            p_glu, p_time, p_length = torch.Tensor(glu[i]).to(self.device), torch.Tensor(time[i]).to(self.device), torch.BoolTensor(get_mask(length[i], self.emb_dim)).to(self.device)
            state, h = self.glu_encoder(p_glu, p_time)
            day_rep = torch.masked_select(state, mask=p_length).reshape((1, len(length[i]), -1))
            o1, h1 = self.encoders(day_rep)
            glu_seq = torch.cat([glu_seq, h1[-1]])
            history_keys.append(torch.squeeze(o1))
        # patient_representations = torch.cat([static_fea, glu_seq], dim=-1)  # (seq, dim*4)
        
        # graph memory module
        if self.ddi_in_memory:
            drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter  # (size, dim)
        else:
            drug_memory = self.ehr_gcn()
        
        drug_memory = drug_memory.to(self.device)
        # static memory
        query = self.query(glu_seq)  
        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (1, size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (batch_size, dim)
        # dynamic memory
        fact2 = []
        for i in range(len(glu)):
            history_med = med[i]

            if len(history_med) > 1:
                history_v = torch.Tensor(history_med[:- 1]).to(self.device)  # (seq-1, dim)
                history_k = history_keys[i][:-1].to(self.device)

                visit_weight = F.softmax(torch.mm(torch.unsqueeze(query[i], 0), history_k.t()))  # (1, seq-1)
                weighted_values = visit_weight.mm(history_v)  # (1, size)
                out = torch.mm(weighted_values, drug_memory)

                fact2.append(out)
            else:

                fact2.append(torch.unsqueeze(fact1[i], 0))

        fact2 = torch.cat(fact2, 0)

        # final representation and generate output
        output = self.output(torch.cat([static_fea, query, fact1, fact2], dim=-1))  # (1, dim)
        
        if self.training:
            neg_pred_prob = F.sigmoid(output)

            neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)  # (voc_size, voc_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()

            return output, batch_neg
        else:
            return output




class MERITS(nn.Module):
    def __init__(self, lab_dim, med_dim, ehr_adj, ddi_adj, glu_dim, time_adj, glu_encoder, emb_dim=64,
                 device=torch.device('cpu:0'),
                 ddi_in_memory=True, dropout=0.4):
        super(MERITS, self).__init__()

        self.lab_dim = lab_dim
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ehr_adj = torch.FloatTensor(ehr_adj).to(device)
        self.tensor_time_adj = torch.FloatTensor(time_adj).to(device)
        self.med_dim = med_dim
        self.ddi_in_memory = ddi_in_memory
        self.emb_dim = emb_dim
        # static featrue

        self.static_ll = nn.Sequential(
            nn.Linear(lab_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(p=dropout)

        # glu_encode
        self.glu_encoder = glu_encoder.to(self.device)
        

        # ATTN_encoder
        self.att_med = Attn_encoder(med_dim, emb_dim, emb_dim // 2)
        self.att_glu = Attn_encoder(glu_dim, emb_dim // 2, emb_dim // 4)

        self.m1_att = self.attention = MultiHeadAttention(emb_dim, 4, dropout)
        self.m2_att = self.attention = MultiHeadAttention(emb_dim, 4, dropout)

        # dynamic_featrue
        self.encoders = nn.GRU(16, emb_dim, batch_first=True)

        self.query = nn.Sequential(

            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )

        # Graph attention
        self.ehr_gat = GraphAttentionLayer(1, emb_dim).to(self.device)
        self.ddi_gat = GraphAttentionLayer(1, emb_dim).to(self.device)
        self.time_gat = GraphAttentionLayer(1, emb_dim).to(self.device)
        self.inter = nn.Parameter(torch.FloatTensor(1)).to(self.device)

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * med_dim, emb_dim * med_dim // 8),
            nn.ReLU(),
            nn.Linear(emb_dim * med_dim // 8, med_dim)
        )

    def forward(self, input, return_graph = False):
        # input (adm, 3, codes)

        # generate medical embeddings and queries
        lab, glu, time, med, length = input

        reps = []
        att1 = []
        att2 = []
        for i in range(len(glu)):
            p_lab, p_glu, p_time, p_length = torch.Tensor(lab[i]).to(self.device), torch.Tensor(glu[i]).to(self.device),  \
                                                        torch.Tensor(time[i]).to(self.device), torch.ByteTensor(get_mask(length[i], self.emb_dim)).to(self.device)
            # static_fea
            
            static_fea = self.static_ll(p_lab).repeat(len(p_length), 1)
            
            # graph_rep
            if len(med[i]) > 1:
                h = torch.unsqueeze(torch.sum(torch.Tensor(med[i])[:-1], 0), -1).to(self.device)
            else:
                h = torch.Tensor([[0] for i in range(self.med_dim)]).to(self.device)
            drug_memmory = self.ehr_gat(h, self.tensor_ehr_adj) + self.time_gat(h, self.tensor_time_adj) - \
                                                                        self.ddi_gat(h, self.tensor_ddi_adj)

            # med_rep
            med_rep = self.att_med(torch.Tensor(med[i])[:-1].to(self.device))

            # glu_rep train = False means the encoder will return fixed-size representation of the glucose
            glu_rep = self.glu_encoder(p_glu, p_time, length[i], train = False)
            
            glu_rep = self.att_glu(glu_rep)
            
            
            # concentrate the static featrue and dynamic feature
            patient_rep = torch.cat((glu_rep, static_fea), -1)

            E_en, att_score1 = self.m1_att(patient_rep, patient_rep, med_rep)
            E_de, att_score2 = self.m2_att(E_en[0], E_en[0], drug_memmory)

            # attention score to visualize
            att1.append(torch.mean(att_score1,0))
            att2.append(torch.mean(att_score2,0))
        
            final_rep = E_de.reshape((1, -1))
            reps.append(final_rep)

        final_rep = torch.cat(reps, 0)

        output = self.output(final_rep)  # (1, dim)

        if self.training:
            neg_pred_prob = F.sigmoid(output)

            neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)  # (med_size, med_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()

            return output, batch_neg
        else:
            if return_graph:
                return output, att1, att2
            else:
                return output



'''
Retain
'''

class Retain(nn.Module):
    def __init__(self, med_dim, glu_encoder, emb_size=64, device=torch.device('cpu:0')):
        super(Retain, self).__init__()

        self.emb_size = emb_size

        self.med_embeddings = torch.nn.Parameter(torch.Tensor(np.random.randn(med_dim, emb_size // 2)))
        self.glu_encoder = glu_encoder
        self.alpha_gru = nn.GRU(emb_size, emb_size, batch_first=True)
        self.beta_gru = nn.GRU(emb_size, emb_size, batch_first=True)

        self.query = nn.Sequential(
            nn.Linear(16 , emb_size // 2),
            nn.ReLU(),
        )

        self.alpha_li = nn.Linear(emb_size, 1)
        self.beta_li = nn.Linear(emb_size, emb_size)

        self.output = nn.Linear(emb_size, med_dim)
        self.med_dim = med_dim
        self.device = device

    def forward(self, input):
        lab, glu, time, med, length = input

        # dynamic featrue
        reps = []

        for i in range(len(glu)):
            p_glu, p_time, p_length = torch.Tensor(glu[i]).to(self.device), torch.Tensor(time[i]).to(self.device), torch.BoolTensor(get_mask(length[i], emb_dim = 16)).to(self.device)

            state, h = self.glu_encoder(p_glu, p_time)
            # print("state:", state.shape)
            # print("p_length",p_length.shape)
            # print(torch.masked_select(state,mask=p_length))
            #
            day_rep = torch.masked_select(state, mask=p_length).reshape((1, len(length[i]), -1))

            # patient_representations = torch.cat([static_fea, glu_seq], dim=-1)  # (seq, dim*4)
            queries = self.query(torch.squeeze(day_rep))  # (T, dim)

            # static

            # input: (visit, 3, codes )
            p_med = med[i].copy()
            p_med[-1] = [0 for j in range(self.med_dim)]

            visit_emb = torch.matmul(torch.Tensor(p_med).to(self.device), (self.med_embeddings))  # (T-1, emb)

            visit_emb = torch.cat([visit_emb, queries], -1)

            g, _ = self.alpha_gru(visit_emb.unsqueeze(dim=0))  # g: (1, T-1, emb)
            h, _ = self.beta_gru(visit_emb.unsqueeze(dim=0))  # h: (1, T-1, emb)

            g = g.squeeze(dim=0)  # (visit - 1, emb)
            h = h.squeeze(dim=0)  # (visit - 1 , emb)
            attn_g = F.softmax(self.alpha_li(g), dim=-1)  # (visit - 1, 1)
            attn_h = F.tanh(self.beta_li(h))  # (visit - 1, emb)

            c = attn_g * attn_h * visit_emb  # (visit - 1, emb)
            c = torch.sum(c, dim=0).unsqueeze(dim=0)  # (1, emb)

            reps.append(c)

        final_rep = torch.cat(reps, 0)

        return self.output(final_rep), None



class MERITS_M(nn.Module):
    def __init__(self, lab_dim, med_dim, ehr_adj, ddi_adj, glu_dim, time_adj, glu_encoder, emb_dim=64,
                 device=torch.device('cpu:0'),
                 ddi_in_memory=True, dropout=0.4):
        super(MERITS_M, self).__init__()

        self.lab_dim = lab_dim
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ehr_adj = torch.FloatTensor(ehr_adj).to(device)
        self.tensor_time_adj = torch.FloatTensor(time_adj).to(device)
        self.med_dim = med_dim
        self.ddi_in_memory = ddi_in_memory
        # static featrue
        self.embeddings = nn.Embedding(lab_dim, emb_dim)
        self.med_embedding = nn.Embedding(med_dim, emb_dim)

        self.static_ll = nn.Sequential(
            nn.Linear(lab_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(p=dropout)

        # glu_encode
        self.glu_encoder = glu_encoder.to(self.device)

        # ATTN_encoder
        self.att_med = Attn_encoder(med_dim, emb_dim, emb_dim // 2)
        self.att_glu = Attn_encoder(glu_dim, emb_dim // 2, emb_dim // 4)

        self.m1_att = self.attention = MultiHeadAttention(emb_dim, 4, dropout)
        self.m2_att = self.attention = MultiHeadAttention(emb_dim, 4, dropout)

        # dynamic_featrue
        self.encoders = nn.GRU(16, emb_dim, batch_first=True)

        self.query = nn.Sequential(

            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )


        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim , emb_dim * med_dim // 8),
            nn.ReLU(),
            nn.Linear(emb_dim * med_dim // 8, med_dim)
        )

    def forward(self, input, return_graph = False):
        # input (adm, 3, codes)

        # generate medical embeddings and queries
        lab, glu, time, med, length = input

        reps = []
        att1 = []
        att2 = []
        for i in range(len(glu)):
            p_lab, p_glu, p_time, p_length = torch.Tensor(lab[i]).to(self.device), torch.Tensor(glu[i]).to(self.device), torch.Tensor(time[i]).to(self.device), torch.BoolTensor(get_mask(length[i])).to(self.device)
            # static_fea

            static_fea = self.static_ll(p_lab).repeat(len(p_length), 1)

            med_rep = self.att_med(torch.Tensor(med[i])[:-1].to(self.device))

            # glu_rep
            
            glu_rep = self.glu_encoder(p_glu, p_time, length[i], train=False)
            
            glu_rep = glu_rep.reshape(len(length[i]), -1)
            glu_rep = self.att_glu(glu_rep)

            patient_rep = torch.cat((glu_rep, static_fea), -1)

            E_en, att_score1 = self.m1_att(patient_rep, patient_rep, med_rep)
            
            final_rep = torch.mean(torch.squeeze(E_en), 0).reshape((1, -1))
            

            reps.append(final_rep)

        final_rep = torch.cat(reps, 0)

        '''R:convert O and predict'''

        output = self.output(final_rep)  # (1, dim)

        if self.training:
            neg_pred_prob = F.sigmoid(output)

            neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)  # (voc_size, voc_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()

            return output, batch_neg
        else:
            if return_graph:
                return output, att1, att2
            else:
                return output


class MERITS_Lin(nn.Module):
    def __init__(self, lab_dim, med_dim, ehr_adj, ddi_adj, glu_dim, time_adj, glu_encoder, emb_dim=64,
                 device=torch.device('cpu:0'),
                 ddi_in_memory=True, dropout=0.4):
        super(MERITS_Lin, self).__init__()

        self.lab_dim = lab_dim
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ehr_adj = torch.FloatTensor(ehr_adj).to(device)
        self.tensor_time_adj = torch.FloatTensor(time_adj).to(device)
        self.med_dim = med_dim
        self.ddi_in_memory = ddi_in_memory
        # static featrue
        self.embeddings = nn.Embedding(lab_dim, emb_dim)
        self.med_embedding = nn.Embedding(med_dim, emb_dim)

        self.static_ll = nn.Sequential(
            nn.Linear(lab_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(p=dropout)

        # glu_encode
        self.glu_encoder = glu_encoder.to(self.device)

        # ATTN_encoder
        self.att_med = Attn_encoder(med_dim, emb_dim, emb_dim // 2)
        self.att_glu = Attn_encoder(glu_dim, emb_dim // 2, emb_dim // 4)

        self.m1_att = MultiHeadAttention(emb_dim, 4, dropout)
        self.m2_att = MultiHeadAttention(emb_dim, 4, dropout)

        self.linlayer1 = nn.Linear(med_dim, emb_dim)
        self.linlayer2 = nn.Linear(med_dim, emb_dim)
        self.linlayer3 = nn.Linear(med_dim, emb_dim)

        # dynamic_featrue
        self.encoders = nn.GRU(16, emb_dim, batch_first=True)

        self.query = nn.Sequential(

            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )

        #
        self.ehr_gat = GraphAttentionLayer(1, emb_dim).to(self.device)
        self.ddi_gat = GraphAttentionLayer(1, emb_dim).to(self.device)
        self.time_gat = GraphAttentionLayer(1, emb_dim).to(self.device)
        self.inter = nn.Parameter(torch.FloatTensor(1)).to(self.device)

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * med_dim, emb_dim * med_dim // 8),
            nn.ReLU(),
            nn.Linear(emb_dim * med_dim // 8, med_dim)
        )

    def forward(self, input):
        # input (adm, 3, codes)

        # generate medical embeddings and queries
        lab, glu, time, med, length = input

        reps = []
        for i in range(len(glu)):
            p_lab, p_glu, p_time, p_length = torch.Tensor(lab[i]).to(self.device), torch.Tensor(glu[i]).to(self.device),  \
                torch.Tensor(time[i]).to(self.device), torch.BoolTensor(get_mask(length[i])).to(self.device)
            # static_fea

            static_fea = self.static_ll(p_lab).repeat(len(p_length), 1)
            # graph_rep
            if len(med[i]) > 1:
                h = torch.unsqueeze(torch.sum(torch.Tensor(med[i])[:-1], 0), -1).to(self.device)
            else:
                h = torch.Tensor([[0] for i in range(self.med_dim)]).to(self.device)

            drug_memmory = self.linlayer1(self.tensor_ehr_adj) + self.linlayer2(self.tensor_time_adj) - self.linlayer3(
                self.tensor_ddi_adj)

            # med_rep

            med_rep = self.att_med(torch.Tensor(med[i])[:-1].to(self.device))

            # glu_rep
            glu_rep = self.glu_encoder(p_glu, p_time, length[i], train=False)

            glu_rep = self.att_glu(glu_rep)

            patient_rep = torch.cat((glu_rep, static_fea), -1)

            E_en, att_score1 = self.m1_att(patient_rep, patient_rep, med_rep)

            E_de, att_score2 = self.m2_att(E_en[0], E_en[0], drug_memmory)

            final_rep = E_de.reshape((1, -1))

            reps.append(final_rep)

        final_rep = torch.cat(reps, 0)

        '''R:convert O and predict'''

        output = self.output(final_rep)  # (1, dim)

        if self.training:
            neg_pred_prob = F.sigmoid(output)

            neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)  # (voc_size, voc_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()

            return output, batch_neg
        else:
            return output

class MERITS_T(nn.Module):
    def __init__(self, lab_dim, med_dim, ehr_adj, ddi_adj, glu_dim, time_adj, glu_encoder, emb_dim=64,
                 device=torch.device('cpu:0'),
                 ddi_in_memory=True, dropout=0.4):
        super(MERITS_T, self).__init__()

        self.lab_dim = lab_dim
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ehr_adj = torch.FloatTensor(ehr_adj).to(device)
        self.tensor_time_adj = torch.FloatTensor(time_adj).to(device)
        self.med_dim = med_dim
        self.ddi_in_memory = ddi_in_memory
        # static featrue
        self.embeddings = nn.Embedding(lab_dim, emb_dim)
        self.med_embedding = nn.Embedding(med_dim, emb_dim)

        self.static_ll = nn.Sequential(
            nn.Linear(lab_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(p=dropout)

        # glu_encode
        self.glu_encoder = glu_encoder

        # ATTN_encoder
        self.att_med = Attn_encoder(med_dim, emb_dim, emb_dim // 2)
        self.att_glu = Attn_encoder(glu_dim, emb_dim // 2, emb_dim // 4)

        self.m1_att = MultiHeadAttention(emb_dim, 4, dropout)
        self.m2_att = MultiHeadAttention(emb_dim, 4, dropout)

        self.graph_att = MultiHeadAttention(emb_dim, 4, dropout)
        # dynamic_featrue
        self.encoders = nn.GRU(16, emb_dim, batch_first=True)

        self.query = nn.Sequential(

            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )

        #
        self.ehr_gat = GraphAttentionLayer(1, emb_dim).to(self.device)
        self.ddi_gat = GraphAttentionLayer(1, emb_dim).to(self.device)
        self.time_gat = GraphAttentionLayer(1, emb_dim).to(self.device)
        self.inter = nn.Parameter(torch.FloatTensor(1)).to(self.device)

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * med_dim, emb_dim * med_dim // 8),
            nn.ReLU(),
            nn.Linear(emb_dim * med_dim // 8, med_dim)
        )

    def forward(self, input):
        # input (adm, 3, codes)

        # generate medical embeddings and queries
        lab, glu, time, med, length = input

        reps = []
        for i in range(len(glu)):
            p_lab, p_glu, p_time, p_length = torch.Tensor(lab[i]).to(self.device), torch.Tensor(glu[i]).to(self.device), torch.Tensor(time[i]).to(self.device), torch.BoolTensor(get_mask(length[i]))
            # static_fea

            static_fea = self.static_ll(p_lab).repeat(len(p_length), 1)
            # graph_rep
            if len(med[i]) > 1:
                h = torch.unsqueeze(torch.sum(torch.Tensor(med[i])[:-1], 0), -1).to(self.device)
            else:
                h = torch.Tensor([[0] for i in range(self.med_dim)]).to(self.device)

            all_graph = torch.cat(
                [self.ehr_gat(h, self.tensor_ehr_adj), self.time_gat(h, self.tensor_time_adj), self.ddi_gat(
                    h, self.tensor_ddi_adj)]).to(self.device)

            drug_memmory, _ = self.graph_att(all_graph, all_graph, all_graph)
            drug_memmory = torch.mean(drug_memmory.reshape(3, self.med_dim, -1), 0)

            # med_rep

            med_rep = self.att_med(torch.Tensor(med[i])[:-1].to(self.device))

            # glu_rep
            glu_rep = self.glu_encoder(p_glu, p_time, length[i], train=False)

            glu_rep = self.att_glu(glu_rep)

            patient_rep = torch.cat((glu_rep, static_fea), -1)

            E_en, att_score1 = self.m1_att(patient_rep, patient_rep, med_rep)

            E_de, att_score2 = self.m2_att(E_en[0], E_en[0], drug_memmory)

            final_rep = E_de.reshape((1, -1))

            reps.append(final_rep)

        final_rep = torch.cat(reps, 0)

        '''R:convert O and predict'''

        output = self.output(final_rep)  # (1, dim)

        if self.training:
            neg_pred_prob = F.sigmoid(output)

            neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)  # (med_size, med_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()

            return output, batch_neg
        else:
            return output

