import torch
from ODE import ODEVAE
import json
import os
import numpy as np


def get_mask(p_length):
    mask_zeors = np.zeros((len(p_length), 25, 16))
    ones = np.ones((1, 16))
    for i in range(len(p_length)):
        mask_zeors[i, p_length[i] - 1, :] = ones
    return mask_zeors


def DataLoader():
    with open(os.getcwd() + "/data_glu_action_lab_basic_diagnose_final.json", "r", ) as f:
        dataset = json.load(f)

    whole_set = []
    for key in dataset:
        data = []
        length = []
        date_last = ''

        for date in dataset[key]:
            glu, time, action, lab = dataset[key][date]
            '''
            glu padding
            '''
            glu = glu[:min(len(glu), 25)]
            time = time[:min(len(time), 25)]
            for i in range(min(len(glu), 25)):
                '''
                nomalize for time : minutes / (24 * 60)
                '''

                time[i] = time[i] / 1440

            temp_length = len(glu)
            for i in range(25 - temp_length):
                time.append(0)
                glu.append(0)
            whole_set.append([glu, time, temp_length])
    return whole_set


dataset = np.array(DataLoader())


vae = ODEVAE(1, 64, 16)
optim = torch.optim.Adam(vae.parameters(), betas=(0.9, 0.999), lr=0.001)

preload = False
batch_size = 32
noise_std = 0.02
n_epochs = int(len(dataset) / batch_size)
model_name = 'ODE'

if preload:
    vae.load_state_dict(torch.load("models/vae_spirals.sd"))
index = 0
data_train = dataset
for epoch_idx in range(n_epochs):
    losses = []
    seq_input = data_train[index:min(index + batch_size, len(data_train))]
    index = (index + batch_size) % len(data_train)
    glu, time, length = [], [], []

    for k in range(len(seq_input)):
        glu.append(seq_input[k][0])
        time.append(seq_input[k][1])
        length.append(seq_input[k][2])

    x_p, z, z_mean, z_log_var = vae(np.array(glu), np.array(time), length)
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), -1)
    loss = 0.5 * (((torch.squeeze(x_p, -1) - torch.Tensor(glu))*torch.Tensor(np.array(glu)>0)) ** 2).sum(-1).sum(0) /\
           noise_std ** 2 + kl_loss

    loss = torch.mean(loss)

    loss /= torch.sum((torch.Tensor(length)))
    loss.backward()
    optim.step()
    losses.append(loss.item())
    print('loss:',0.5 * (((torch.squeeze(x_p, -1) - torch.Tensor(glu))*torch.Tensor(np.array(glu)>0)) ** 2).sum(-1).sum(0))
    print("Epoch {}, Loss {}".format(x_p.detach().numpy()[0][:length[0]], glu[0][:length[0]]))
    print("Epoch {}, Loss {}".format(epoch_idx, loss))

torch.save(vae.state_dict(), open(
    os.path.join('saved', model_name,
                 'ode_encoder.model'), 'wb+'))
