def row_normal(data, factor=1e6):
    row_sum = np.sum(data, axis=1)
    row_sum = np.expand_dims(row_sum, 1)
    div = np.divide(data, row_sum)
    div = np.log(1 + factor * div)
    return div


def load_newdata(train_datapath, metric='pearson', data_type='count', trans=True):
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
    data_min = np.min(df.values, axis=0).reshape([1, df.shape[1]])
    data_max = np.max(df.values, axis=0).reshape([1, df.shape[1]])
    minmax = np.append(data_min, data_max, axis=0)
    minmax_df = pd.DataFrame(data=minmax, index=['min', 'max'])
    minmax_path = "{}/minmax_{}.csv".format(outdir, name)
    if os.path.exists(minmax_path) is False:
        minmax_df.to_csv(minmax_path,index=True)
    if gene_scale:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data = scaler.fit_transform(df)
        df = pd.DataFrame(data=data, columns=df.columns)
    return df.values


def weighted_mse_pre(y_true, y_pred):
    ''' Hack: This fn doesn't accept additional arguments.
              We use y_true to pass them.
        y_pred: Contains x_hat - x
        y_true: Contains [b, deg]
    '''
    return K.sum(
        K.square(y_pred)* y_true,
        axis=-1)


class Autoencoder():
    def __int__(self):
        pass

    def pretrain(self):
        X_train_tmp = train_set
        self.trained_encoders = []
        self.trained_decoders = []
        for i in range(len(dims) - 1):
            print('Pre-training the layer: Input {} -> {} -> Output {}'.format(dims[i], dims[i + 1], dims[i]))
            # Create AE and training
            ae = Sequential()
            if i == 0:
                print(i)
                if i == 0:
                    x = Input(shape=(dims[0],), name='input')
                    x_drop = Dropout(dr_rate)(x)
                    h = Dense(dims[i + 1], input_dim=dims[i], activation='relu',
                              W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                              name='encoder_%d' % i)(x_drop)
                    y = Dense(dims[i], input_dim=dims[i + 1], W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                              name='decoder_%d' % i)(h)
                    x_diff = Subtract()([x, y])
                    ae = Model(inputs=x, outputs=x_diff)
                    ae.compile(loss=weighted_mse_pre, optimizer='adam')
                    ae.fit(x = train_set, y= B, batch_size=batch_size, epochs=epochs_pretrain)
                    ae.summary()
                    # Store trainined weight
                    self.trained_encoders.append(ae.layers[2])
                    self.trained_decoders.append(ae.layers[3])
                    # Update training data
                    encoder = Model(ae.input, ae.layers[2].output)
                    X_train_tmp = encoder.predict(X_train_tmp)
            else:
                if i == len(dims) - 2:
                    ae.add(Dropout(dr_rate))
                    ae.add(Dense(dims[i + 1], input_dim=dims[i], W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                                 name='encoder_%d' % i))
                    ae.add(Dense(dims[i], input_dim=dims[i + 1], activation='relu', W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                                 name='decoder_%d' % i))
                else:
                    ae.add(Dropout(dr_rate))
                    ae.add(Dense(dims[i + 1], input_dim=dims[i], activation='relu', W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                                 name='encoder_%d' % i))
                    ae.add(Dense(dims[i], input_dim=dims[i + 1], activation='relu', W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                                 name='decoder_%d' % i))
                ae.compile(loss='mean_squared_error', optimizer='adam')
                ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, epochs=epochs_pretrain)
                ae.summary()
                # Store trainined weight
                self.trained_encoders.append(ae.layers[1])
                self.trained_decoders.append(ae.layers[2])
                # Update training data
                encoder = Model(ae.input, ae.layers[1].output)
                X_train_tmp = encoder.predict(X_train_tmp)

    def ae_build(self):
        print('Fine-tuning')
        self.decoders = Sequential()
        self.encoders = Sequential()
        # autoencoders.add(InputLayer(input_shape=(dims[0],)))
        # encoders.add(InputLayer(input_shape=(dims[0],)))
        for encoder in self.trained_encoders:
            #     autoencoders.add(encoder)
            self.encoders.add(encoder)
        for decoder in self.trained_decoders[::-1]:
            self.decoders.add(decoder)
        self.x = Input(shape=(dims[0],), name='input')
        self.h = self.encoders(self.x)
        self.y = self.decoders(self.h)
        self.x_diff = Subtract()([self.x, self.y])
        self.ae = Model(inputs=self.x, outputs=self.x_diff)
        self.autoencoders = Model(inputs=self.x, outputs=self.y)


    def train_ae(self):
        self.ae.compile(optimizer='adam', loss=weighted_mse_pre)
        checkpoint = ModelCheckpoint(filepath=outdir + 'best_ae.h5', monitor='loss', save_best_only=True,
                                     save_weights_only=True)
        self.history = self.ae.fit(x = train_set, y= B, batch_size=batch_size,
                                             nb_epoch=epochs_ae, callbacks=[checkpoint])
        # self.ae.save_weights('{}/sdae.h5'.format(outdir))


    def load_pretrain_weights(self):
        self.autoencoders.load_weights('{}/autoencoder_pretrain.h5'.format(outdir))


    def save_imputation(self, sta):
        mask_data = train_set == 0.0
        mask_data = np.float32(mask_data)
        decoder_out = self.autoencoders.predict(train_set)
        decoder_out_replace = mask_data * decoder_out + train_set
        # df_raw = pd.DataFrame(decoder_out)
        # df_raw.to_csv('{}/sdae_{}.csv'.format(outdir, sta), index=None, float_format='%.4f')
        df_replace = pd.DataFrame(decoder_out_replace)
        df_replace.to_csv('{}/sdae_r_{}.csv'.format(outdir, sta), index=None, float_format='%.4f')


    def plot_loss(self):
        f = plt.figure()
        plt.plot(self.history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()
        f.savefig("{}/{}_ae.png".format(outdir, name), bbox_inches='tight')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--n_iters_ae', default=2000, type=int)
    parser.add_argument('--n_iters_pretrain', default=1000, type=int)
    parser.add_argument('--beta1', default=0.1, type=float)
    parser.add_argument('--beta2', default=1.0, type=float)
    parser.add_argument('--alpha', default=0.01, type=float)
    parser.add_argument('--dr_rate', default=0.2, type=float)
    parser.add_argument('--nu1', default=0.0, type=float)
    parser.add_argument('--nu2', default=0.0, type=float)
    parser.add_argument("--train_datapath", default="/home/xysmlx/data/filter_data/zeisel_count.csv", type=str)
    parser.add_argument("--data_type", default="count", type=str)
    parser.add_argument("--outDir", default="/home/xysmlx/data/", type=str)
    parser.add_argument("--name", default="zeisel", type=str)
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--gene_scale', dest='gene_scale', action='store_true')
    feature_parser.add_argument('--no-gene_scale', dest='gene_scale', action='store_false')
    parser.set_defaults(gene_scale=False)
    parser.add_argument('--GPU_SET', default="3", type=str)



    args = parser.parse_args()
    print(args)

    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_SET

    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import keras

    import keras.backend.tensorflow_backend as KTF

    KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))
    from keras.engine.topology import Layer, InputSpec
    from keras.layers import Input, Dense, Lambda, Subtract, merge, Dropout, BatchNormalization, Activation
    from keras.models import Model, model_from_json, Sequential
    import keras.regularizers as Reg
    from keras.optimizers import SGD, Adam
    from keras import backend as K

    from keras.callbacks import ModelCheckpoint

    import math
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    plt.switch_backend('agg')
    from time import time
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    name = args.name
    train_datapath = args.train_datapath
    batch_size = args.batch_size
    n_iters_pretrain = args.n_iters_pretrain
    n_iters_ae = args.n_iters_ae
    dr_rate = args.dr_rate
    nu1 = args.nu1
    nu2 = args.nu2
    outdir = args.outDir
    beta1 = args.beta1
    beta2 = args.beta2
    alpha = args.alpha
    gene_scale = args.gene_scale

    train_set = load_newdata(train_datapath, data_type=args.data_type)
    nsamples = len(train_set)
    steps_per_epoch = nsamples // batch_size
    if nsamples < batch_size:
        steps_per_epoch = 1
        batch_size = nsamples
    B = np.ones(train_set.shape) * beta1
    B[train_set != 0] = beta2

    dims = [train_set.shape[1], 500, 500, 2000, 10]

    epochs_pretrain = max(n_iters_pretrain // steps_per_epoch, 1)
    epochs_ae = max(n_iters_ae // steps_per_epoch, 1)
    ae = Autoencoder()
    ae.pretrain()
    ae.ae_build()
    ae.train_ae()
    ae.save_imputation('ae')
    ae.plot_loss()

