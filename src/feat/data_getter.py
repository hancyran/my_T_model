import os
import numpy as np
import pandas as pd

from src.feat.ad_data_getter import getAdData
from src.feat.user_data_getter import getUserData
from src.utils.path_args import args


def getData(train_type):
    if train_type == 'cv':
        X_train, Y_train, test_df = getAdData('cv')
#         X_train_user = getUserData('cv')
#         X_train = np.concatenate((X_train, X_train_user), axis=1)
        return X_train, Y_train, test_df
    elif train_type == 'val':
        X_train, Y_train, X_test, Y_test, test_df = getAdData('val')
#         X_train_user, X_test_user = getUserData('val')
#         X_train = np.concatenate((X_train, X_train_user), axis=1)
#         X_test = np.concatenate((X_test, X_test_user), axis=1)
        return X_train, Y_train, X_test, Y_test, test_df
    elif train_type == 'test':
        X_train, Y_train, X_test, test_df = getAdData('test')
#         X_train_user, X_test_user = getUserData('test')
#         X_train = np.concatenate((X_train, X_train_user), axis=1)
#         X_test = np.concatenate((X_test, X_test_user), axis=1)
        return X_train, Y_train, X_test, test_df
    else:
        raise Exception('Error while loading data')


def getFinalData(train_type):
    if train_type == 'cv':
        if os.path.exists(args.tmp_data_path+'/tmp_train_final_data.npy'):
            X_train = np.load(args.tmp_data_path+'/tmp_train_final_data.npy')
            _, Y_train, test_df = getAdData('cv')
        else:
            X_train, Y_train, test_df = getAdData('cv')
            X_train_user = getUserData('cv')
            X_train = np.concatenate((X_train, X_train_user), axis=1)
        return X_train, Y_train, test_df
    elif train_type == 'val':
        if os.path.exists(args.tmp_data_path+'/tmp_tri_train_final_data.npy') and os.path.exists(args.tmp_data_path+ '/tmp_tri_train_final_data.npy'):
            X_train = np.load(args.tmp_data_path+'/tmp_tri_train_final_data.npy')
            X_test = np.load(args.tmp_data_path+'/tmp_tri_train_final_data.npy')
            _, Y_train, _, Y_test, test_df = getAdData('val')

        else:
            X_train, Y_train, X_test, Y_test, test_df = getAdData('val')
            X_train_user, X_test_user = getUserData('val')
            X_train = np.concatenate((X_train, X_train_user), axis=1)
            X_test = np.concatenate((X_test, X_test_user), axis=1)
        return X_train, Y_train, X_test, Y_test, test_df
    elif train_type == 'test':
        if os.path.exists(args.tmp_data_path+'/tmp_train_final_data.npy'):
            X_train = np.load(args.tmp_data_path+'/tmp_train_final_data.npy')
            X_test = np.load(args.tmp_data_path+'/tmp_test_final_data.npy')
            _, Y_train, _, test_df = getAdData('test')
        else:
            X_train, Y_train, X_test, test_df = getAdData('test')
            X_train_user, X_test_user = getUserData('test')
            X_train = np.concatenate((X_train, X_train_user), axis=1)
            X_test = np.concatenate((X_test, X_test_user), axis=1)
        return X_train, Y_train, X_test, test_df
    else:
        raise Exception('Error while loading data')


def getData_feat2(train_type):
    if train_type == 'cv':
        X_train = pd.read_hdf(args.tmp_data_path+'/tmp_train_feat2_ad.h5').values
        X_train = np.delete(X_train, 7, axis=1)
        Y_train = np.load(args.train_data_path+'/train_arr_label.npy')
        test_df = pd.read_hdf(args.train_data_path+'/train_origin_final.h5')
        return X_train, Y_train, test_df
    elif train_type == 'val':
        X_train = pd.read_hdf(args.tmp_data_path+'/tmp_train_feat2_ad.h5').values
        X_train = np.delete(X_train, 7, axis=1)
        Y_train = np.load(args.train_data_path+'/train_arr_label.npy')
        test_df = pd.read_hdf(args.train_data_path+'/train_origin_final.h5')
        # pepare for dataset
        i = test_df.loc[test_df.日期 == 319].index
        m = test_df.loc[test_df.日期 != 319].index
        ### train dataset
        X_tri_data = X_train[m]
        Y_tri_label = Y_train[m]
        ### test dataset
        X_val_data = X_train[i]
        Y_val_label = Y_train[i]
        return X_tri_data, Y_tri_label, X_val_data, Y_val_label, test_df
    elif train_type == 'test':
        X_train = pd.read_hdf(args.tmp_data_path+'/tmp_train_feat2_ad.h5').values
        X_train = np.delete(X_train, 7, axis=1)
        Y_train = np.load('train2/train_arr_label.npy')
        test_df = pd.read_hdf('test2/test_origin_final.h5')

        X_test = pd.read_hdf('tmp/tmp_test_feat2_ad.h5').values

        return X_train, Y_train, X_test, test_df
    else:
        raise Exception('No such train type')


def getData_feat3(train_type):
    if train_type == 'cv':
        X_train = np.load(args.tmp_data_path+'/tmp_train_feat3_withlen.npy')
        Y_train = np.load(args.train_data_path+'/train_arr_label.npy')
        test_df = pd.read_hdf(args.train_data_path+'/train_origin_final.h5')
        return X_train, Y_train, test_df
    elif train_type == 'val':
        X_train = np.load(args.tmp_data_path+'/tmp_train_feat3_withlen.npy')
        Y_train = np.load(args.train_data_path+'/train_arr_label.npy')
        test_df = pd.read_hdf(args.train_data_path+'/train_origin_final.h5')
        # pepare for dataset
        i = test_df.loc[test_df.日期 == 319].index
        m = test_df.loc[test_df.日期 != 319].index
        ### train dataset
        X_tri_data = X_train[m]
        Y_tri_label = Y_train[m]
        ### test dataset
        X_val_data = X_train[i]
        Y_val_label = Y_train[i]
        return X_tri_data, Y_tri_label, X_val_data, Y_val_label, test_df
    elif train_type == 'test':
        X_train = np.load(args.tmp_data_path+'/tmp_train_feat3_withlen.npy')
        Y_train = np.load(args.train_data_path+'/train_arr_label.npy')
        test_df = pd.read_hdf(args.test_data_path+'/test_origin_final.h5')

        X_test = np.load(args.tmp_data_path+'/tmp_test_feat3_withlen.npy')

        return X_train, Y_train, X_test, test_df
    else:
        raise Exception('No such train type')