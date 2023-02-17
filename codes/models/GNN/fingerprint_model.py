# import torch
# from torch_geometric.data import DataLoader
# import torch.optim as optim
# import torch.nn.functional as F
# from ogb.lsc import PCQM4Mv2Dataset, PCQM4Mv2Evaluator
# from ogb.utils import smiles2graph
# from torch_geometric.data import Data
# from gnn import GNN
# import os
# from tqdm import tqdm
# import argparse
# import time
# import numpy as np
# import random


# class Dataset_From_Smiles(object):
#     def __init__(self, smiles_list, smiles2graph=smiles2graph):
#         super(Dataset_From_Smiles, self).__init__()
#         self.smiles_list = smiles_list 
#         self.smiles2graph = smiles2graph

#     def __getitem__(self, idx):
#         '''Get datapoint with index'''
#         data = Data()
#         smiles= self.smiles_list[idx]
#         try:
#             graph = self.smiles2graph(smiles)
#         except:
#             print(smiles)

#         data.__num_nodes__ = int(graph['num_nodes'])
#         data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
#         data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
#         data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)

#         return data

#     def __len__(self):
#         '''Length of the dataset
#         Returns
#         -------
#         int
#             Length of Dataset
#         '''
#         return len(self.smiles_list)

# def get_fingerprint_GNN(smiles,model_name, checkpoint, device, batch_size=32):
#     print('sfdsdsdsdsdsdsdssdsd')


#     if model_name == 'gin':
#         model = GNN(gnn_type = 'gin', num_layers = 5, emb_dim = 600, drop_ratio =0, virtual_node = False, graph_pooling='sum').to(device)
#     elif model_name == 'gin-virtual':
#         model = GNN(gnn_type = 'gin', num_layers = 5, emb_dim = 600, drop_ratio =0, virtual_node = True, graph_pooling='sum').to(device)
#     elif model_name == 'gcn':
#         model = GNN(gnn_type = 'gcn', num_layers = 5, emb_dim = 600, drop_ratio =0, virtual_node = False, graph_pooling='sum').to(device)
#     elif model_name == 'gcn-virtual':
#         model = GNN(gnn_type = 'gcn', num_layers = 5, emb_dim = 600, drop_ratio =0, virtual_node = True, graph_pooling='sum').to(device)

#     model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model_state_dict'])

#     model = model.to(device)
#     model.eval()
#     data =Dataset_From_Smiles(smiles)
#     loader = DataLoader(data, batch_size=batch_size, shuffle=False)

#     embd = []

#     model.eval()
#     for step, batch in enumerate(tqdm(loader, desc="Iteration")):
#         batch = batch.to(device)

#         with torch.no_grad():
#             layer_rep = []
#             node, all_layer= model.gnn_node(batch, all_layer=True)
#             for num, layer in enumerate(all_layer):
#                 rep = model.pool(layer, batch.batch)
#                 layer_rep.append(rep.detach().cpu())

#             embd.append(torch.stack(layer_rep))
#     return torch.cat(embd, dim=1) 



import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from ogb.lsc import PCQM4Mv2Dataset, PCQM4Mv2Evaluator
from ogb.utils import smiles2graph
from torch_geometric.data import Data
from gnn import GNN
import os
from tqdm import tqdm
import argparse
import time
import numpy as np
import random
import pdb

class Dataset_From_Smiles(object):
    def __init__(self, smiles_list, smiles2graph=smiles2graph):
        super(Dataset_From_Smiles, self).__init__()
        self.smiles_list = smiles_list 
        self.smiles2graph = smiles2graph

    def __getitem__(self, idx):
        '''Get datapoint with index'''
        data = Data()
        smiles= self.smiles_list[idx]
        graph = self.smiles2graph(smiles)

        data.__num_nodes__ = int(graph['num_nodes'])
        data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
        data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
        data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
        
        return data

    def __len__(self):
        '''Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        '''
        return len(self.smiles_list)

def get_fingerprint_GNN(smiles,model_name, checkpoint=None, device='cuda', batch_size=32, feature= 'full', checkpoint_type=False):
    
    if model_name == 'gin':
        model = GNN(gnn_type = 'gin', num_layers = 5, emb_dim = 600, drop_ratio =0, virtual_node = False, graph_pooling='sum')
    elif model_name == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_layers = 5, emb_dim = 600, drop_ratio =0, virtual_node = True, graph_pooling='sum')
    elif model_name == 'gcn':
        model = GNN(gnn_type = 'gcn', num_layers = 5, emb_dim = 600, drop_ratio =0, virtual_node = False, graph_pooling='sum')
    elif model_name == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_layers = 5, emb_dim = 600, drop_ratio =0, virtual_node = True, graph_pooling='sum')

    if checkpoint!=None:
        print(checkpoint)
        if checkpoint_type:
            model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model_state_dict'])
        


    model = model.to(device)
    model = model.eval()
    data =Dataset_From_Smiles(smiles)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    embd = []

    model.eval()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            layer_rep = []
            if feature == 'simple':
                    batch.x = batch.x[:,:1]
            node, all_layer= model.gnn_node(batch, all_layer=True)
            for num, layer in enumerate(all_layer):
                rep = model.pool(layer, batch.batch)
                layer_rep.append(rep.detach().cpu())

            embd.append(torch.stack(layer_rep))
    return torch.cat(embd, dim=1) 



