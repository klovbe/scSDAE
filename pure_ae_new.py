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
        minmax_df.to_csv(minmax_path, index=True)
    if gene_scale:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data = scaler.fit_transform(df)
        df = pd.DataFrame(data=data, columns=df.columns)
    return df.values


def batch_generator(X, batch_size, shuffle, beta=1.0, alpha=1.0):
    sample_index = np.arange(X.shape[0])
    number_of_batches = X.shape[0] // batch_size
    counter = 0
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = \
            sample_index[batch_size * counter: batch_size * (counter + 1)]
        X_batch = X[batch_index, :]
        InData = X_batch

        B = np.ones(X_batch.shape)
        B_0 = np.ones(X_batch.shape)
        deg_0 = np.sum(B[X_batch == 0])
        deg_0 = np.max([deg_0, 1])
        deg_0 = np.ones([X_batch.shape[0], 1]) * deg_0
        deg = np.sum(B[X_batch != 0])
        deg = np.max([deg, 1])
        deg = np.ones([X_batch.shape[0], 1]) * deg
        B[X_batch != 0] = beta
        B[X_batch == 0] = 0.0
        B_0[X_batch == 0] = alpha
        B_0[X_batch != 0] = 0.0
        B_0 = np.append(B_0, deg_0, axis=1)
        B = np.append(B, deg, axis=1)
        OutData = [B, B_0]
        counter += 1
        yield InData, OutData
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def weighted_mse(y_true, y_pred):
    ''' Hack: This fn doesn't accept additional arguments.
              We use y_true to pass them.
        y_pred: Contains x_hat - x
        y_true: Contains [b, deg]
    '''
    min_batch_size = K.cast(K.shape(y_true)[0], 'float32')
    return K.sum(K.square(y_pred) * y_true[:, :-1], axis=-1) / y_true[:, -1] * min_batch_size


def weighted_mae(y_true, y_pred):
    ''' Hack: This fn doesn't accept additional arguments.
              We use y_true to pass them.
        y_pred: Contains x_hat - x
        y_true: Contains [b, deg]
    '''
    min_batch_size = K.cast(K.shape(y_true)[0], 'float32')
    return K.sum(K.abs(y_pred) * y_true[:, :-1], axis=-1) / y_true[:, -1] * min_batch_size


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
            print(i)
            if i == 0:
                x = Input(shape=(dims[0],), name='input')
                x_drop = Dropout(dr_rate)(x)
                h = Dense(dims[i + 1], input_dim=dims[i], activation='relu', W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                             name='encoder_%d' % i)(x_drop)
                y = Dense(dims[i], input_dim=dims[i + 1], W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                             name='decoder_%d' % i)(h)
                x_diff1 = Subtract()([x, y])
                x_diff2 = Subtract()([x, y])
                ae = Model(inputs=x, outputs=[x_diff1, x_diff2])
                ae.compile(loss=[weighted_mse, weighted_mae], optimizer='adam')
                ae.fit_generator(batch_generator(train_set, batch_size=batch_size, shuffle=True, beta=1.0, alpha=alpha)
                                        , steps_per_epoch=steps_per_epoch, nb_epoch=epochs_pretrain)
                ae.summary()
                # Store trainined weight
                self.trained_encoders.append(ae.layers[2])
                self.trained_decoders.append(ae.layers[3])
                # Update training data
                encoder = Model(ae.input, ae.layers[2].output)
                X_train_tmp = encoder.predict(X_train_tmp)
            elif i == len(dims) - 2:
                ae = Sequential()
                ae.add(Dropout(dr_rate))
                ae.add(Dense(dims[i + 1], input_dim=dims[i], W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                             name='encoder_%d' % i))
                ae.add(Dense(dims[i], input_dim=dims[i + 1], activation='relu', W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                             name='decoder_%d' % i))
                ae.compile(loss='mean_squared_error', optimizer='adam')
                ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, nb_epoch=epochs_pretrain)
                ae.summary()
                # Store trainined weight
                self.trained_encoders.append(ae.layers[1])
                self.trained_decoders.append(ae.layers[2])
                # Update training data
                encoder = Model(ae.input, ae.layers[2].output)
                X_train_tmp = encoder.predict(X_train_tmp)
            else:
                ae = Sequential()
                ae.add(Dropout(dr_rate))
                ae.add(Dense(dims[i + 1], input_dim=dims[i], activation='relu', W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                             name='encoder_%d' % i))
                ae.add(Dense(dims[i], input_dim=dims[i + 1], activation='relu', W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                             name='decoder_%d' % i))
                ae.compile(loss='mean_squared_error', optimizer='adam')
                ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, nb_epoch=epochs_pretrain)
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
        self.x_diff1 = Subtract()([self.x, self.y])
        self.x_diff2 = Subtract()([self.x, self.y])
        # self.x_diffa = K.abs(self.x_dff)
        # self.x_diffs = K.square(self.x_dff)
        self.autoencoders = Model(inputs=self.x, outputs=[self.x_diff1, self.x_diff2])
        self.autoencoders.summary()
        self.predict_model = Model(inputs=self.x, outputs=self.y)


    def simple_ae_build(self, h_dim):
        print('only one layer autoencoder')
        self.x = Input(shape=(dims[0],), name='input')
        self.x_drop = Dropout(dr_rate)(self.x)
        self.h = Dense(h_dim, input_dim=train_set.shape[1], activation='relu', W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                  name='encoder')(self.x_drop)
        self.y = Dense(train_set.shape[1], input_dim=h_dim, W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2),
                  name='decoder')(self.h)
        self.x_diff1 = Subtract()([self.x, self.y])
        self.x_diff2 = Subtract()([self.x, self.y])
        self.autoencoders = Model(inputs=self.x, outputs=[self.x_diff1, self.x_diff2])
        self.autoencoders.summary()
        self.encoder = Model(self.x, self.h)
        self.autoencoders = Model(inputs=self.x, outputs=[self.x_diff1, self.x_diff2])
        self.autoencoders.summary()
        self.predict_model = Model(inputs=self.x, outputs=self.y)


    def train_ae(self):
        self.autoencoders.compile(optimizer='adam', loss=[weighted_mse, weighted_mae])
        checkpoint = ModelCheckpoint(filepath=outdir + 'best.h5', monitor='loss', save_best_only=True,
                                     save_weights_only=True)
        self.history = self.autoencoders.fit_generator(batch_generator(train_set, batch_size=batch_size
                                                                       , shuffle=True, beta=1.0, alpha=alpha),
                                                       steps_per_epoch=steps_per_epoch,
                                                       epochs=epochs_ae, callbacks=[checkpoint])
        # self.autoencoders.save_weights('{}/autoencoder_pure.h5'.format(outdir))

    def load_pretrain_weights(self):
        self.autoencoders.load_weights('{}/autoencoder_pretrain.h5'.format(outdir))

    def save_imputation(self):
        mask_data = train_set == 0.0
        mask_data = np.float32(mask_data)
        decoder_out = self.predict_model.predict(train_set)
        decoder_out_replace = mask_data * decoder_out + train_set
        # df_raw = pd.DataFrame(decoder_out)
        # df_raw.to_csv('{}/autoencoder.csv'.format(outdir), index=None, float_format='%.4f')
        df_replace = pd.DataFrame(decoder_out_replace)
        df_replace.to_csv('{}/autoencoder_r.csv'.format(outdir), index=None, float_format='%.4f')

    def plot_loss(self):
        f = plt.figure()
        plt.plot(self.history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()
        f.savefig("{}/{}.png".format(outdir, name), bbox_inches='tight')
        f = plt.figure()
        plt.plot(self.history.history['subtract_3_loss'])
        plt.title('mse loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()
        f.savefig("{}/{}_mse.png".format(outdir, name), bbox_inches='tight')
        f = plt.figure()
        plt.plot(self.history.history['subtract_4_loss'])
        plt.title('mae loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()
        f.savefig("{}/{}_mae.png".format(outdir, name), bbox_inches='tight')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--n_iters_ae', default=2000, type=int)
    parser.add_argument('--n_iters_pretrain', default=1000, type=int)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--dr_rate', default=0.2, type=float)
    parser.add_argument('--nu1', default=0.0, type=float)
    parser.add_argument('--nu2', default=0.0, type=float)
    parser.add_argument("--train_datapath", default="/data/wlchi/data/filter_data/zeisel_count.csv", type=str)
    parser.add_argument("--data_type", default="count", type=str)
    parser.add_argument("--outDir", default="/data/wlchi/python_project/SSDAE", type=str)
    parser.add_argument("--name", default="SmartSeq_count", type=str)
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
    import os
    from keras.layers import Input, Dense, Lambda, Subtract, merge, Dropout, BatchNormalization, Activation
    from keras.models import Model, model_from_json, Sequential
    import keras.regularizers as Reg
    from keras.optimizers import SGD, Adam
    from keras import backend as K

    from keras.callbacks import ModelCheckpoint
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
    alpha = args.alpha
    gene_scale = args.gene_scale

    train_set = load_newdata(train_datapath, data_type=args.data_type)
    nsamples = len(train_set)
    steps_per_epoch = nsamples // batch_size
    if nsamples < batch_size:
        steps_per_epoch = 1
        batch_size = nsamples

    dims = [train_set.shape[1], 500, 500, 2000, 10]

    epochs_pretrain = max(n_iters_pretrain // steps_per_epoch, 1)
    epochs_ae = max(n_iters_ae // steps_per_epoch, 1)

    optimizer = SGD(lr=0.1, momentum=0.99)

    ae = Autoencoder()
    ae.pretrain()
    ae.ae_build()
    ae.train_ae()
    ae.save_imputation()
    ae.plot_loss()
