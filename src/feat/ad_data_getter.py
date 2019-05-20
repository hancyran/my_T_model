import os
import gc

import numpy as np
import pandas as pd


def getAdData(train_type):
    if train_type == 'cv':
        #### train dataset
        if os.path.exists('tmp/tmp_train_data.npy'):
            X_train = np.load('tmp/tmp_train_data.npy')
        else:
            X_train = np.load('train2/train_arr_create_weekday.npy')
            X_train = np.concatenate((X_train, np.load('train2/train_arr_ad_factory_id.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_product_type_id.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_product_id.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_acc_id.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_period.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_numeric.npy')), axis=1)
            # save as tmp file
            np.save('tmp/tmp_train_data.npy', X_train)

        # train label
        Y_train = np.load('train2/train_arr_label.npy')

        # train_df
        test_df = pd.read_hdf('train/train_origin_final.h5')

        return X_train, Y_train, test_df

    elif train_type == 'val':
        # load tmp file if exists
        if os.path.exists('tmp/tmp_tri_train_data.npy') and os.path.exists(
                'tmp/tmp_tri_train_label.npy') and os.path.exists(
                'tmp/tmp_val_train_data.npy') and os.path.exists('tmp/tmp_val_train_label.npy'):
            X_tri_data = np.load('tmp/tmp_tri_train_data.npy')
            Y_tri_label = np.load('tmp/tmp_tri_train_label.npy')
            X_val_data = np.load('tmp/tmp_val_train_data.npy')
            Y_val_label = np.load('tmp/tmp_val_train_label.npy')

            # train_df
            test_df = pd.read_hdf('train/train_origin_final.h5')

        else:
            #### original train dataset
            if os.path.exists('tmp/tmp_train_data.npy'):
                X_train = np.load('tmp/tmp_train_data.npy')
            else:
                X_train = np.load('train2/train_arr_create_weekday.npy')
                X_train = np.concatenate((X_train, np.load('train2/train_arr_ad_factory_id.npy')), axis=1)
                X_train = np.concatenate((X_train, np.load('train2/train_arr_product_type_id.npy')), axis=1)
                X_train = np.concatenate((X_train, np.load('train2/train_arr_product_id.npy')), axis=1)
                X_train = np.concatenate((X_train, np.load('train2/train_arr_acc_id.npy')), axis=1)
                X_train = np.concatenate((X_train, np.load('train2/train_arr_period.npy')), axis=1)
                X_train = np.concatenate((X_train, np.load('train2/train_arr_numeric.npy')), axis=1)
                # save as tmp file
                np.save('tmp/tmp_train_data.npy', X_train)

            # original train label
            Y_train = np.load('train2/train_arr_label.npy')

            # train_df
            test_df = pd.read_hdf('train/train_origin_final.h5')

            # pepare for dataset
            i = test_df.loc[test_df.日期 == 319].index
            m = test_df.loc[test_df.日期 != 319].index
            ### train dataset
            X_tri_data = X_train[m]
            Y_tri_label = Y_train[m]
            np.save('tmp/tmp_tri_train_data.npy', X_tri_data)
            np.save('tmp/tmp_tri_train_label.npy', Y_tri_label)
            ### test dataset
            X_val_data = X_train[i]
            Y_val_label = Y_train[i]
            np.save('tmp/tmp_val_train_data.npy', X_val_data)
            np.save('tmp/tmp_val_train_label.npy', Y_val_label)
            ## collect
            del X_train, Y_train
            gc.collect()

        return X_tri_data, Y_tri_label, X_val_data, Y_val_label, test_df

    ###### todo : assign the corresponding dataset for test
    elif train_type == 'test':
        #### original rain dataset
        if os.path.exists('tmp/tmp_train_data.npy'):
            X_train = np.load('tmp/tmp_train_data.npy')
        else:
            X_train = np.load('train2/train_arr_create_weekday.npy')
            X_train = np.concatenate((X_train, np.load('train2/train_arr_ad_factory_id.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_product_type_id.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_product_id.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_acc_id.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_period.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_numeric.npy')), axis=1)
            # save as tmp file
            np.save('tmp/tmp_train_data.npy', X_train)

        # original train label
        Y_train = np.load('train2/train_arr_label.npy')

        # train_df
        test_df = pd.read_hdf('test2/test_origin_final.h5')

        ### test dataset
        if os.path.exists('tmp/tmp_test_data.npy'):
            X_test = np.load('tmp/tmp_test_data.npy')
        else:
            X_test = np.load('test2/test_arr_create_weekday.npy')
            X_test = np.concatenate((X_test, np.load('test2/test_arr_ad_factory_id.npy')), axis=1)
            X_test = np.concatenate((X_test, np.load('test2/test_arr_product_type_id.npy')), axis=1)
            X_test = np.concatenate((X_test, np.load('test2/test_arr_product_id.npy')), axis=1)
            X_test = np.concatenate((X_test, np.load('test2/test_arr_acc_id.npy')), axis=1)
            X_test = np.concatenate((X_test, np.load('test2/test_arr_period.npy')), axis=1)
            X_test = np.concatenate((X_test, np.load('test2/test_arr_numeric.npy')), axis=1)
            # save as tmp file
            np.save('tmp/tmp_test_data.npy', X_test)
        ###

        return X_train, Y_train, X_test, test_df

    else:
        raise Exception('No Such Train Type')
