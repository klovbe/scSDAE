import numpy as np
import scanpy.api as sc
from dca.api import dca
import pandas as pd
import h5py
import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'   #指定第一块GPU可用
from tqdm import *
# import fire

def dca_reconstruct(path, save_path, data_type):
    print("make dataset from {}...".format(path))
    df = pd.read_csv(path, sep=",", index_col=0)
    df = df.transpose()
    # if data_type == 'logcount':
    #     x = (df != 0).astype('float')
    #     df = x * np.exp(df)
    if data_type == 'logcount':
        df = np.exp(df) - 1
    print("have {} samples, {} features".format(df.shape[0], df.shape[1]))
    adata_ae = sc.AnnData(df.values.astype('int'), var=df.columns)
    adata_ae.var_names = df.columns
    sc.pp.filter_genes(adata_ae, min_counts=1)
    dca(adata_ae, threads=1, hidden_size=[500, 500, 2000, 10, 2000, 500, 500], batch_size=256, verbose=True,
        batchnorm=False, epochs=1000)
    data = pd.DataFrame(data=adata_ae.X, index=df.index, columns=adata_ae.var_names)
    data.to_csv(save_path)

def dca_recon_dir(dir, save_dir, filters=None):
    name_list = os.listdir(dir)
    name_list_new = []
    for file in name_list:
        if "patel_logcount.csv" in file:
            name_list_new.append(file)
    name_list = name_list_new
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    for name in tqdm(name_list):
        name1 = name.split(".")[0].split("_")[0]
        data_type = name.split(".")[0].split("_")[1]
        # print(data_type)
        # if name1 not in ["kolod", "klein", "zeisel", "manno", "baron", "chen", "shekhar", "campbell", "havrin"]:
#        if name1 not in [ "biase","camp, "darmanis","deng", "goolam", "havrin",  "macosko", "pollen", "treutlein", "usoskin", "yan"]:
        path = os.path.join(dir, name)
        save_path = os.path.join(save_dir, name)
        dca_reconstruct(path, save_path, data_type)

if __name__ == "__main__":
    # fire.Fire()
    # magic_reconstruct("./filter_data/baron_count.csv", "./magic_filter_data/baron.csv")
    # magic_reconstruct("/home/xysmlx/data/run_data/cite.csv", "/home/xysmlx/python_project/magic/cite.csv")
    # magic_recon_dir("/home/xysmlx/data/filter_data/", "/home/xysmlx/python_project/magic/")
    # for i in range(1, 11):
        # dca_recon_dir("/data/wlchi/missingdata/elegans/down/{}/".format(i), "/data/wlchi/missingdata/elegans/dca/{}/".format(i))
    # fire.Fire()
    # dca_recon_dir("/data/wlchi/missingdata/elegans", "/data/wlchi/missingdata/elegans/dca")
    # dca_reconstruct("/data/wlchi/missingdata/cite_count.csv", "/data/wlchi/missingdata/cite_dca.csv", 'count')
    dca_recon_dir("/data/wlchi/data/filter_data", "/data/wlchi/missingdata/cluster/dca")
    # dca_recon_dir("/data/wlchi/data/filter_data/down", "/data/wlchi/missingdata/cluster/dca/down")