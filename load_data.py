import numpy as np
import pandas as pd
import scanpy.api as sc


def row_normal(data, factor=1e6):
    row_sum = np.sum(data, axis=1)
    row_sum = np.expand_dims(row_sum, 1)
    div = np.divide(data, row_sum)
    div = np.log(1 + factor * div)
    return div

def load_newdata(train_datapath, metric='pearson', gene_scale=False, data_type='count', trans=True ):
    print("make dataset from {}...".format(train_datapath))
    df = pd.read_csv(train_datapath, sep=",", index_col=0)
    if trans:
        df = df.transpose()
    print("have {} samples, {} features".format(df.shape[0], df.shape[1]))
    if data_type == 'count':
        df = row_normal(df)
        # df = sizefactor(df)
    elif data_type == 'rpkm':
        df = np.log(df + 1)
    if gene_scale:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data = scaler.fit_transform(df)
        df = pd.DataFrame(data=data, columns=df.columns)
    return df.values

def extract_features(data, gene_select=10000):
    # sehng xu pai lie qu zuida de ruo gan ji yin, ran hou dao xu
    selected = np.std(data, axis=0)
    selected =selected.argsort()[-gene_select:][::-1]
    h_data = data[:,selected]
    return h_data

def load_data_scanpy(train_datapath, data_type='count', trans=True):
    print("make dataset from {}...".format(train_datapath))
    df = pd.read_csv(train_datapath, sep=",", index_col=0)
    if trans:
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
    # sc.pp.highly_variable_genes(adata, n_top_genes=20000, flavor='cell_ranger', inplace=True)
    # adata = adata[:, adata.var['highly_variable']]
    # if gene_scale:
    #     sc.pp.scale(adata, zero_center=True, max_value=3)
    return adata.X

