"""
MOSAIC Training Module
GCN autoencoder training for multi-view dimensionality reduction.
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from models import init_model_dict, init_optim
from utils import gen_adj_mat_tensor, knbrsloss
from os.path import splitext, basename, isfile

cuda = True if torch.cuda.is_available() else False


def prepare_trte_data(file_input, dataset_name):
    """Load multi-view data from file list, cache as .fea files."""
    sample_names = []
    l = []
    data_tr_list = []
    tmp_dir = './fea/' + dataset_name + '/'
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)

    for line in open(file_input, 'rt'):
        base_file = splitext(basename(line.rstrip()))[0]
        fea_save_file = tmp_dir + base_file + '.fea'
        if isfile(fea_save_file):
            df_new = pd.read_csv(fea_save_file, sep=',', header=0, index_col=0)
            l = list(df_new)
        df_new = df_new.T
        sample_names = df_new.index.tolist()
        data_tr_list.append(df_new.values.astype(float))

    data_tensor_list = [torch.FloatTensor(d) for d in data_tr_list]
    return data_tensor_list, l, sample_names


def gen_trte_adj_mat(data_tr_list, num_class=6):
    """Generate adjacency matrices for each view."""
    adj_metric = "euclidean"
    adj_train_list = [
        gen_adj_mat_tensor(data_tr_list[i], num_class, adj_metric)
        for i in range(len(data_tr_list))
    ]
    return adj_train_list


def train_epoch(data_list, adj_list, model_dict, optim_dict, k, pre_flag):
    """Single training epoch: contrastive loss + reconstruction loss."""
    torch.autograd.set_detect_anomaly(True)
    loss_dict = {}
    embedding_list = []
    output_list2 = []
    mean_emb = []

    criterion = torch.nn.MSELoss()
    for m in model_dict:
        model_dict[m].train()
    num_view = len(data_list)

    # ---- Inter-view + Intra-view Contrastive Learning ----
    if pre_flag is True:
        for i in range(num_view):
            optim_dict["G{:}".format(i + 1)].zero_grad()
            embedding_list.append(
                model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i])
            )

        stack_tensor_list2 = torch.stack(embedding_list)
        mean_clu = stack_tensor_list2.mean(dim=0)

        for i in range(num_view):
            ci_loss = criterion(
                embedding_list[i],
                torch.nn.Parameter(mean_clu, requires_grad=False),
            )
            gi_loss = knbrsloss(embedding_list[i], k)
            total_ccloss = ci_loss + gi_loss
            total_ccloss.backward()
            optim_dict["G{:}".format(i + 1)].step()
            loss_dict["G{:}".format(i + 1)] = (
                gi_loss.detach().cpu().numpy().item()
            )

    # ---- Decoder Reconstruction ----
    if num_view >= 2:
        for i in range(num_view):
            optim_dict["M{:}".format(i + 1)].zero_grad()
            embedding_list[i] = model_dict["E{:}".format(i + 1)](
                data_list[i], adj_list[i]
            )

        stack_tensor_list = torch.stack(embedding_list)
        mean_emb = torch.mean(stack_tensor_list, dim=0)

        for i in range(num_view):
            output_list2.append(
                model_dict["M{:}".format(i + 1)](
                    torch.nn.Parameter(mean_emb, requires_grad=False)
                )
            )
            m_loss = criterion(output_list2[i], data_list[i])
            m_loss.backward(retain_graph=True)
            optim_dict["M{:}".format(i + 1)].step()
            loss_dict["M{:}".format(i + 1)] = (
                m_loss.detach().cpu().numpy().item()
            )

    return loss_dict, embedding_list, mean_emb


def train_test(
    file_input, data_folder, view_list, num_class, lr_e, total_epochs
):
    """
    Full GCN autoencoder training pipeline.

    Returns
    -------
    mean_emb : torch.Tensor
        Mean embedding across all views.
    l : list
        Feature names from the data.
    """
    num_view = len(view_list)
    dim_he_list = [300, 200, 100]

    data_tr_list, l, sample_names = prepare_trte_data(file_input, data_folder)
    adj_tr_list = gen_trte_adj_mat(data_tr_list, num_class)
    k = int(data_tr_list[0].shape[0] / num_class + 1)
    dim_list = [x.shape[1] for x in data_tr_list]

    model_dict = init_model_dict(num_view, dim_list, dim_he_list, num_class)

    embedding_list = []
    mean_emb = []

    for m in model_dict:
        if cuda:
            model_dict[m].cuda()
    for m in range(num_view):
        if cuda:
            data_tr_list[m] = data_tr_list[m].cuda()
            adj_tr_list[m] = adj_tr_list[m].cuda()

    print("\nTraining ...\n")
    loss_list_dict = {}

    optim_dict1 = init_optim(num_view, model_dict, lr_e)
    for epoch in range(total_epochs + 1):
        loss_dict, embedding_list, mean_emb = train_epoch(
            data_tr_list, adj_tr_list, model_dict, optim_dict1, k, True
        )
        for key in loss_dict:
            loss_list_dict.setdefault(key, []).append(loss_dict[key])

        if epoch == total_epochs:
            print("Epoch {}: loss_dict = {}".format(epoch, loss_dict))

    # Save loss curves
    for key in loss_list_dict:
        plt.plot(loss_list_dict[key])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig('./loss_png/' + data_folder + ('{}_show.png'.format(key)))
    plt.close('all')

    # Save per-view embeddings
    for view_idx in range(num_view):
        view_emb = embedding_list[view_idx].detach().cpu().numpy()
        view_fea = pd.DataFrame(
            view_emb,
            index=sample_names,
            columns=[f'Dim_{i}' for i in range(view_emb.shape[1])],
        ).T
        view_fea.to_csv(f'./fea/_view{view_idx + 1}.csv')
        print(f'view{view_idx + 1} feature saved')

    return mean_emb, l
