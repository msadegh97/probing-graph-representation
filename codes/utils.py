import numpy as np
import scipy.sparse as sp
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import log_loss
from sklearn.neighbors import kneighbors_graph

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, log_loss, accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import PoissonRegressor

import deepchem as dc
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

from joblib import Memory
memory = Memory("cache_dir")

@memory.cache
def compute_properties(smiles):
    desc_names = [k for (k, v) in Descriptors.descList]
    calculator = MolecularDescriptorCalculator(desc_names)
    properties = []
    for smile in tqdm(smiles):
        m = Chem.MolFromSmiles(smile)
        properties.append(calculator.CalcDescriptors(m))
    properties = pd.DataFrame(properties, columns=desc_names)

    return desc_names, properties

def nnz_average(x):
    """
    Compute the average of only the "non-zero" elements in the sparse matrix x.
    """
    x = x.tocsr()
    sums = x.sum(axis=1).A1
    counts = np.diff(x.indptr)
    
    averages = np.zeros_like(sums, dtype='float')
    averages[counts > 0] = sums[counts > 0]/counts[counts > 0]

    return averages

# def __local_purity(adj, property, kind):
#     frequencies = sp.diags(1/adj.sum(1).A1) @ adj @ np.eye(property.max() + 1)[property]
#     if kind == 'max':
#         purity = frequencies.max(axis=1)
#     elif kind == 'entropy':
#         purity = -(frequencies*np.log(frequencies+1e-18)).sum(1)
#     else:
#         raise NotImplementedError
#     return np.mean(purity)

# def local_purity(adj_list, property, kind='max'):
#     return np.array([__local_purity(adj=adj, property=property, kind=kind) for adj in adj_list])

def __local_variance(adj, prop):
    adj_var = adj.copy()
    adj_var.data = np.abs(prop[adj_var.indices] - np.repeat(prop, np.diff(adj_var.indptr)))
    return np.mean(nnz_average(adj_var))

def local_variance(adj_list, prop):
    return np.array([__local_variance(adj=adj, prop=prop) for adj in adj_list])

def __knn_probe(adj, prop):
    adj_var = adj.copy()
    adj_var.data = prop[adj_var.indices]
#     return ((nnz_average(adj_var)>0.5) == prop).mean()
    return roc_auc_score(prop, nnz_average(adj_var))

def knn_probe(adj_list, prop):
    return np.array([__knn_probe(adj, prop) for adj in adj_list])

# @numba.jit(nopython=True)
def _top_k(indices, indptr, data, min_k_per_row):
    """
    Parameters
    ----------
    indices: np.ndarray, shape [n_edges]
        Indices of a sparse matrix.
    indptr: np.ndarray, shape [n+1]
        Index pointers of a sparse matrix.
    data: np.ndarray, shape [n_edges]
        Data of a sparse matrix.
    min_k_per_row: np.ndarray, shape [n]
        Number of top_k elements for each row.
    Returns
    -------
    top_k_idx: list
        List of the indices of the top_k elements for each row.
    """
    n = len(indptr) - 1
    top_k_idx = []
    for i in range(n):
        cur_top_k = min_k_per_row[i]
        if cur_top_k > 0:
            cur_indices = indices[indptr[i]:indptr[i + 1]]
            cur_data = data[indptr[i]:indptr[i + 1]]
            top_k = cur_indices[cur_data.argsort()[-cur_top_k:]]
            top_k_idx.append(top_k)

    return top_k_idx


def top_k_per_row(x, k_per_row):
    """
    Returns the indices of the top_k element per row for a sparse matrix.
    Considers only the non-zero entries.
    Parameters
    ----------
    x : sp.spmatrix, shape [n, n]
        Data matrix.
    k_per_row : np.ndarray, shape [n]
        Number of top_k elements for each row.
    Returns
    -------
    top_k_per_row : np.ndarray, shape [?, 2]
        The 2D indices of the top_k elements per row.
    """
    # make sure that k_per_row does not exceed the number of entries per row
    counts = np.diff(x.indptr)
    min_k_per_row = np.minimum(k_per_row, counts)
    # min_k_per_row = np.minimum(k_per_row, (x != 0).sum(1).A1)

    # if len(np.unique(counts)) == 1 and len(np.unique(min_k_per_row)) == 1 and optimize:
    #     k = min_k_per_row[0]
    #     print(counts[0])
    #     idxs = (np.arange(k) + (np.arange(x.shape[0])*counts[0])[:, None]).reshape(-1)
    #     return sp.csr_matrix((np.ones_like(idxs), x.indices[idxs], np.arange(x.shape[0]+1)*k), dtype='int')
    
    n = x.shape[0]
    row_idx = np.repeat(np.arange(n), min_k_per_row)

    col_idx = _top_k(x.indices, x.indptr, x.data, min_k_per_row)
    col_idx = np.concatenate(col_idx)

    top_k_per_row = np.column_stack((row_idx, col_idx))

    return top_k_per_row

def edges_to_sparse(edges, shape, weights=None):
    """Create a sparse adjacency matrix from an array of edge indices and (optionally) values.
    Parameters
    ----------
    edges : array-like, shape [num_edges, 2]
        Array with each row storing indices of an edge as (u, v).
    shape : (int, int)
        Shape of the sparse matrix
    weights : array_like, shape [num_edges], optional, default None
        Weights of the edges. If None, all edges weights are set to 1.
    Returns
    -------
    A : sp.csr_matrix
        Adjacency matrix in CSR format.
    """
    if weights is None:
        weights = np.ones(edges.shape[0], dtype='int')

    return sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=shape).tocsr()

def get_knn_graphs(embedding, max_size_cluster = 1000, num_samples = 10, spacing='linear'):
    # adj = [kneighbors_graph(embd[:max_entity], nn + 1, include_self=True, n_jobs=8) for nn in nns]

    adj_all = kneighbors_graph(embedding, max_size_cluster + 1, include_self=True, n_jobs=8, mode='distance')

    if spacing == 'linear':
        num_neighbors = np.linspace(2, max_size_cluster, num_samples).astype(np.int32)
    elif spacing == 'log':
        num_neighbors = np.geomspace(2, max_size_cluster, num_samples).astype(np.int32)
    else:
        raise NotImplementedError
    
    adj_list = [edges_to_sparse(top_k_per_row(-adj_all, nn + 1), adj_all.shape) for nn in tqdm(num_neighbors)]

    assert np.allclose(np.array([np.unique(tmp.sum(1).A1) for tmp in adj_list]).T , (num_neighbors+1))

    return num_neighbors, adj_list


def evaluate_properties(num_neighbors, adj_list, properties, probing_func=local_variance, binarize=True, standardize=True, just_discrete=False):
    curves = []
    evaluated_properties = []
    aucs = []

    for property_name in tqdm(properties.columns):    
        cur_property = properties[property_name].values.copy()
        is_discrete = cur_property.dtype == np.dtype('int')

        if np.isnan(cur_property).any() or np.isclose(cur_property.var(), 0):
            continue

        if is_discrete:
            if binarize:
                cur_property[cur_property>1] = 1

            values, counts = np.unique(cur_property, return_counts=True)
        
            if counts.max()/len(properties) > 0.96 or 0 not in values:
                continue
        else:
            if just_discrete:
                continue
            if standardize:
                cur_property = (cur_property-cur_property.min())/(cur_property.max()-cur_property.min())
                if cur_property.var() < 0.01:
                    continue

        curve = probing_func(adj_list=adj_list, prop=cur_property)
        curves.append(curve)
        
        prop_auc = auc(num_neighbors, curve)
        
        aucs.append(prop_auc)
        evaluated_properties.append(property_name)
                                
    curves = np.array(curves)
    aucs = np.array(aucs)
    evaluated_properties = np.array(evaluated_properties)

    return curves, aucs, evaluated_properties




def valid_smiles(smiles):
    valid_idx=[]
    for i, smi in enumerate(smiles):
        m = Chem.MolFromSmiles(smi,sanitize=True)
        if m !=None:
            valid_idx.append(i)

    return valid_idx



def linear_probing(embedding_train,y_train,  embeding_test=None, y_test= None, seed=0, percent_train=0.8, scale=True, **kwargs):
    np.random.seed(seed)
    if (type(embeding_test) == type(None)):
        n_train = int(percent_train*len(y_train))
        rnd = np.random.permutation(len(y_train))
        idx_train, idx_test = rnd[:n_train], rnd[n_train:]
      
        train = embedding_train[idx_train]
        test = embedding_train[idx_test]
        y_test = y_train[idx_test]
        y_train = y_train[idx_train]
    else:
        train = embedding_train
        test = embeding_test
   
    mask_train =  (y_train == y_train).reshape(-1)
    mask_test =  (y_test == y_test).reshape(-1)
    
    train = train[mask_train]
    test = test[mask_test]
    y_train = y_train[mask_train]
    y_test = y_test[mask_test]
    
    if scale:
        scaler = preprocessing.StandardScaler()
        train = scaler.fit_transform(train)
        test = scaler.transform(test)
    if kwargs['task'] == 'classification':
            
        model = LogisticRegression(verbose=False, max_iter=2000)
#         model = MLPClassifier(hidden_layer_sizes=300,)
        model.fit(train, y_train)
        y_proba = model.predict_proba(test)
        y_pred = y_proba.argmax(1)
        res = {"log_loss":log_loss(y_test, y_proba),
               "accuracy": accuracy_score(y_pred, y_test),
               "f1_score":f1_score(y_pred, y_test, average='macro'),
               "roc_auc": roc_auc_score(y_test, y_proba[:, 1], multi_class='ovr')}
        
    elif kwargs['task'] == 'PoissonRegressor':
        model = PoissonRegressor()
        model.fit(train, y_train)
        y_pred = model.predict(test)
        res = {"MSE":mean_squared_error(y_test, y_pred),
               "R2":r2_score(y_test, y_pred),
                "MAE": mean_absolute_error(y_test,y_pred)}
    elif kwargs['task'] == 'regression':
        model = LinearRegression()
        model.fit(train, y_train)
        y_pred = model.predict(test)
        res = {"MSE":mean_squared_error(y_test, y_pred),
               "R2":r2_score(y_test, y_pred),
                "MAE": mean_absolute_error(y_test,y_pred)}

    
    return res