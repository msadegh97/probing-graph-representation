#!/usr/bin/env python
# coding: utf-8


import sys
import os
from argparse import Namespace
import deepchem as dc
from rdkit import Chem
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, log_loss, accuracy_score, f1_score
from ogb.lsc import PCQM4Mv2Dataset
import pickle

current = os.path.dirname(os.path.realpath('__file__'))
parent = os.path.dirname(current)
sys.path.append(parent)
from utils.utils import *
from utils.splitters import ScaffoldSplitter
from fingerprint_model import get_fingerprint_GNN


# In[35]:

if __name__ == "__main__":

    root='../dataset'
    gnns = ['gcn',]


    # In[36]:


    import pickle 
    with open('../pcqmv2_subset_dataset.pkl', 'rb') as handle:
        ids = pickle.load(handle)
    train_ids = ids['train_ids_100k']

    with open('../pcqmv2_valid_subset_dataset.pickle', 'rb') as handle:
        ids_valid = pickle.load(handle)
    valid_ids = ids_valid['valid_train_ids_100k']


    # In[39]:


    dataset = PCQM4Mv2Dataset(root=root, only_smiles=True)
    splits= dataset.get_idx_split()
    all_data = pd.read_csv(os.path.join(root,'pcqm4m-v2/raw/data.csv.gz'))
    smiles_train = all_data.iloc[train_ids].smiles.values.tolist() 
    smiles_test= all_data.iloc[valid_ids].smiles.values.tolist() 
    
    desc_names_tarin, properties_train = compute_properties(smiles_train)
    desc_names_test, properties_test = compute_properties(smiles_test)


    selected_prop = ['NumSaturatedRings',]
    #         'NumAromaticRings',
    #         'NumAromaticCarbocycles',
    #         'fr_aniline',
    #         'fr_ketone_Topliss',
    #         'fr_ketone',
    #         'fr_bicyclic',
    #         'fr_methoxy',
    #         'fr_para_hydroxylation',
    #         'fr_pyridine',
    #         'fr_benzene']

    y_train = properties_train[selected_prop].values.copy()
    y_train[y_train>1] = 1 # binarize

    y_test = properties_test[selected_prop].values.copy()
    y_test[y_test>1] = 1 # binarize


    # In[ ]:


    # np.save('./valid_pcqm_GCN_virtual',embd[-1].numpy())


    # In[ ]:


    res_all_mean={}
    res_all_std={}
    for gnn in gnns:

        for random_init in [True]:#, False]:
            if random_init:
                embd_train = get_fingerprint_GNN(smiles_train,f'{gnn}-virtual', checkpoint_type=True, batch_size=512)
                embd_test = get_fingerprint_GNN(smiles_test,f'{gnn}-virtual', checkpoint_type=True, batch_size=512)
            else:
                embd_train = get_fingerprint_GNN(smiles_train,f'{gnn}-virtual', checkpoint=f'../../checkpoints/{gnn}_virtual/checkpoint.pt',batch_size=512)
                embd_test = get_fingerprint_GNN(smiles_test,f'{gnn}-virtual', checkpoint=f'../../checkpoints/{gnn}_virtual/checkpoint.pt',batch_size=512)

            
            
            
                
            res=[]
            for n_label in range(y_train.shape[1]):  
                if len(np.unique(y_test[:,n_label][(y_test[:,n_label] == y_test[:,n_label])])) < 2:
                    continue
                res.append(linear_probing(embedding_train=embd_train[-1], y_train=y_train[:,n_label],
                                        embeding_test=embd_test[-1], y_test=y_test[:, n_label], task='classification',scale=True,))
            con = pd.DataFrame(res, index=selected_prop).T
            res_all_mean[f'{gnn}{"_random" if random_init else ""}']= con


    # In[17]:

    print(pd.concat(res_all_mean)[selected_prop] * 100)

