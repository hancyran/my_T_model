import os
import shutil

import numpy as np
import pandas as pd

from src.feat.ad_data_getter import getAdData
from src.feat.user_data_getter import getUserData

from src.utils.path_args import pargs



def getDfmData(train_type):
    if train_type == 'cv':
        if os.path.exists(pargs.tmp_data_path + '/tmp_train_final_data.npy'):
            X_train = np.load(pargs.tmp_data_path + '/tmp_train_final_data.npy')
            Y_train = np.load(pargs.train_data_path + '/train_arr_label.npy')

            shutil.copy(pargs.train_data_path + '/train_origin_final.h5', './')
            test_df = pd.read_hdf('train_origin_final.h5')
            # _, Y_train, test_df = getAdData('cv')
        else:
            X_train, Y_train, test_df = getAdData('cv')
            X_train_user = getUserData('cv')
            X_train = np.concatenate((X_train, X_train_user), axis=1)
        return X_train, Y_train, test_df

    elif train_type == 'val':
        if os.path.exists(pargs.tmp_data_path + '/tmp_tri_train_final_data.npy') \
                and os.path.exists(pargs.tmp_data_path + '/tmp_tri_train_final_data.npy'):
            X_train = np.load(pargs.tmp_data_path + '/tmp_tri_train_final_data.npy')
            X_test = np.load(pargs.tmp_data_path + '/tmp_tri_train_final_data.npy')
            label = np.load(pargs.train_data_path + '/train_arr_label.npy')

            shutil.copy(pargs.train_data_path + '/train_origin_final.h5', './')
            test_df = pd.read_hdf('train_origin_final.h5')
            i = test_df.loc[test_df.日期 == 319].index
            m = test_df.loc[test_df.日期 != 319].index

            Y_train = label[m]
            Y_test = label[i]
            # _, Y_train, _, Y_test, test_df = getAdData('val')
        else:
            X_train, Y_train, X_test, Y_test, test_df = getAdData('val')
            X_train_user, X_test_user = getUserData('val')
            X_train = np.concatenate((X_train, X_train_user), axis=1)
            X_test = np.concatenate((X_test, X_test_user), axis=1)
        return X_train, Y_train, X_test, Y_test, test_df

    elif train_type == 'test':
        if os.path.exists(pargs.tmp_data_path + '/tmp_train_final_data.npy'):
            X_train = np.load(pargs.tmp_data_path + '/tmp_train_final_data.npy')
            X_test = np.load(pargs.tmp_data_path + '/tmp_test_final_data.npy')

            Y_train = np.load(pargs.train_data_path + '/train_arr_label.npy')

            shutil.copy(pargs.test_data_path + '/test_origin_final.h5', './')
            test_df = pd.read_hdf('test_origin_final.h5')
            # _, Y_train, _, test_df = getAdData('test')
        else:
            X_train, Y_train, X_test, test_df = getAdData('test')
            X_train_user, X_test_user = getUserData('test')
            X_train = np.concatenate((X_train, X_train_user), axis=1)
            X_test = np.concatenate((X_test, X_test_user), axis=1)
        return X_train, Y_train, X_test, test_df

    else:
        raise Exception('Error while loading data')
