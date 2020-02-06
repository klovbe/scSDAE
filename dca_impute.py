import numpy as np
import scanpy.api as sc
from dca.api import dca
import pandas as pd
import h5py
import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'   #指定第一块GPU可用
from metrics import *
from sklearn.cluster import KMeans


# train_datapath = sys.argv[1]
# output_path = sys.argv[2]
dataset_name = 'chen'
data_type = 'count'
n_features = 500
train_datapath = '/data/wlchi/data/filter_data/{}_{}.csv'.format(dataset_name, data_type)
print("make dataset from {}...".format(train_datapath))
df = pd.read_csv(train_datapath, sep=",", index_col=0)
df = df.transpose()
print("have {} samples, {} features".format(df.shape[0], df.shape[1]))
adata = sc.AnnData(df.values)
sc.pp.filter_cells(adata, min_genes=1)
sc.pp.filter_genes(adata, min_cells=1)
if data_type == 'count':
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
elif data_type == 'rpkm':
    sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=n_features, flavor='cell_ranger', inplace=True)
adata_ae = sc.AnnData(df.values)
adata_ae = adata_ae[:, adata.var['highly_variable']]
# sc.pp.highly_variable_genes(adata, n_top_genes=20000, flavor='cell_ranger', inplace=True)
labelpath = '/data/wlchi/data/filter_label/{}_label.csv'.format(dataset_name)
from sklearn.preprocessing import LabelEncoder
labeldf = pd.read_csv(labelpath, header=0, index_col=0)
label = labeldf.values
label = label.transpose()
y_name = np.squeeze(label)
if not isinstance(label, (int, float)):
    label = LabelEncoder().fit_transform(y_name)
n_clusters = len(np.unique(label))



# train_datapath = "/home/xysmlx/data/ercc/CELseq2_count.csv"
# output_path = "/home/xysmlx/data/dca_out/CELseq2_count.csv"
# print(train_datapath)
# print(output_path)
# counts = pd.read_csv(train_datapath, index_col=0)
# counts = counts.transpose()
# adata = sc.AnnData(counts.values)
# sc.pp.filter_genes(adata, min_counts=1)
# sc.pp.normalize_per_cell(adata)
# sc.pp.log1p(adata)

dca(adata_ae, threads=1, hidden_size=[256, 64, 32, 64, 256], batch_size=256, mode='latent', verbose=True, batchnorm=False, epochs=1000)
# data = pd.DataFrame(data=adata.X, index=counts.index, columns=counts.columns)
# data.to_csv(output_path)
pd.DataFrame(adata_ae.obsm['X_dca'], index=None, columns=None).to_csv('./{}_latent.tsv'.format(dataset_name), sep='\t',
                                                                  float_format='%.6f')

latent = adata_ae.obsm['X_dca']
kmeans = KMeans(n_clusters=n_clusters, n_init=40)
y_pred = kmeans.fit_predict(latent)
acc_ = np.round(acc(label, y_pred), 5)
nmi_ = np.round(nmi(label, y_pred), 5)
ari_ = np.round(ari(label, y_pred), 5)
print(ari_)
