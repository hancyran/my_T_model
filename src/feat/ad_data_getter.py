import os
import gc

import numpy as np
import pandas as pd

from src.utils.path_args import pargs
from src.utils.feat_args import fargs


def getAdData(train_type):
    if train_type == 'cv':
        #### train dataset
        if os.path.exists(pargs.tmp_data_path + '/tmp_train_data.npy'):
            X_train = np.load(pargs.tmp_data_path + '/tmp_train_data.npy')
        else:
            X_train = np.load(pargs.train_data_path + '/train_arr_id.npy')
            for n, i in enumerate(fargs.ad_feat):
                if n == 0:
                    pass
                elif n < 7:
                    X_train = np.concatenate((X_train, np.load(pargs.train_data_path + '/train_arr_%s.npy' % i)), axis=1)
                else:
                    X_train = np.concatenate((X_train, np.load(pargs.train_data_path + '/train_arr_%s_scaled.npy' % i)),
                                             axis=1)
            # # save as tmp file
            np.save(pargs.tmp_data_path + '/tmp_train_data.npy', X_train)

        # train label
        Y_train = np.load(pargs.train_data_path + '/train_arr_label.npy')

        # train_df
        test_df = pd.read_hdf(pargs.train_data_path + '/train_origin_final.h5')

        return X_train, Y_train, test_df

    elif train_type == 'val':
        # load tmp file if exists
        if os.path.exists(pargs.tmp_data_path + '/tmp_tri_train_data.npy') \
                and os.path.exists(pargs.tmp_data_path + '/tmp_tri_train_label.npy') \
                and os.path.exists(pargs.tmp_data_path + '/tmp_val_train_data.npy') \
                and os.path.exists(pargs.tmp_data_path + '/tmp_val_train_label.npy'):
            X_tri_data = np.load(pargs.tmp_data_path + '/tmp_tri_train_data.npy')
            Y_tri_label = np.load(pargs.tmp_data_path + '/tmp_tri_train_label.npy')
            X_val_data = np.load(pargs.tmp_data_path + '/tmp_val_train_data.npy')
            Y_val_label = np.load(pargs.tmp_data_path + '/tmp_val_train_label.npy')

            # train_df
            test_df = pd.read_hdf(pargs.train_data_path + '/train_origin_final.h5')

        else:
            #### original train dataset
            if os.path.exists(pargs.tmp_data_path + '/tmp_train_data.npy'):
                X_train = np.load(pargs.tmp_data_path + '/tmp_train_data.npy')
            else:
                X_train = np.load(pargs.train_data_path + '/train_arr_id.npy')
                for n, i in enumerate(fargs.ad_feat):
                    if n == 0:
                        pass
                    elif n < 7:
                        X_train = np.concatenate((X_train, np.load(pargs.train_data_path + '/train_arr_%s.npy' % i)), axis=1)
                    else:
                        X_train = np.concatenate((X_train, np.load(pargs.train_data_path + '/train_arr_%s_scaled.npy' % i)),
                                                 axis=1)

                # save as tmp file
                np.save(pargs.tmp_data_path + '/tmp_train_data.npy', X_train)

            # original train label
            Y_train = np.load(pargs.train_data_path + '/train_arr_label.npy')
            # train_df
            test_df = pd.read_hdf(pargs.train_data_path + '/train_origin_final.h5')

            # pepare for dataset
            i = test_df.loc[test_df.日期 == 319].index
            m = test_df.loc[test_df.日期 != 319].index
            ### train dataset
            X_tri_data = X_train[m]
            Y_tri_label = Y_train[m]
            # np.save(args.tmp_data_path + '/tmp_tri_train_data.npy', X_tri_data)
            # np.save(args.tmp_data_path + '/tmp_tri_train_label.npy', Y_tri_label)
            ### test dataset
            X_val_data = X_train[i]
            Y_val_label = Y_train[i]
            # np.save(args.tmp_data_path + '/tmp_val_train_data.npy', X_val_data)
            # np.save(args.tmp_data_path + '/tmp_val_train_label.npy', Y_val_label)
            ## collect
            del X_train, Y_train
            gc.collect()

        return X_tri_data, Y_tri_label, X_val_data, Y_val_label, test_df

    elif train_type == 'test':
        #### original rain dataset
        if os.path.exists(pargs.tmp_data_path + '/tmp_train_data.npy'):
            X_train = np.load(pargs.tmp_data_path + '/tmp_train_data.npy')
        else:
            if os.path.exists(pargs.tmp_data_path + '/tmp_train_data.npy'):
                X_train = np.load(pargs.tmp_data_path + '/tmp_train_data.npy')
            else:
                X_train = np.load(pargs.train_data_path + '/train_arr_id.npy')
                for n, i in enumerate(fargs.ad_feat):
                    if n == 0:
                        pass
                    elif n < 7:
                        X_train = np.concatenate((X_train, np.load(pargs.train_data_path + '/train_arr_%s.npy' % i)), axis=1)
                    else:
                        X_train = np.concatenate((X_train, np.load(pargs.train_data_path + '/train_arr_%s_scaled.npy' % i)),
                                                 axis=1)

                # save as tmp file
                np.save(pargs.tmp_data_path + '/tmp_train_data.npy', X_train)

        # original train label
        Y_train = np.load(pargs.train_data_path + '/train_arr_label.npy')

        # train_df
        test_df = pd.read_hdf(pargs.test_data_path + '/test_origin_final.h5')

        ### test dataset
        if os.path.exists(pargs.tmp_data_path + '/tmp_test_data.npy'):
            X_test = np.load(pargs.tmp_data_path + '/tmp_test_data.npy')
        else:
            X_test = np.load(pargs.test_data_path + '/test_arr_id.npy')
            for n, i in enumerate(fargs.ad_feat):
                if n == 0:
                    pass
                elif n < 7:
                    X_test = np.concatenate((X_test, np.load(pargs.test_data_path + '/test_arr_%s.npy' % i)), axis=1)
                else:
                    X_test = np.concatenate((X_test, np.load(pargs.test_data_path + '/test_arr_%s_scaled.npy' % i)),
                                            axis=1)

            np.save(pargs.tmp_data_path + '/tmp_test_data.npy', X_test)

        return X_train, Y_train, X_test, test_df

    else:
        raise Exception('No Such Train Type')
