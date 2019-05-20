import os
import numpy as np


from src.feat.ad_data_getter import getAdData
from src.feat.user_data_getter import getUserData


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
        if os.path.exists('tmp/tmp_train_final_data.npy'):
            X_train = np.load('tmp/tmp_train_final_data.npy')
            _, Y_train, test_df = getAdData('cv')
        else:
            X_train, Y_train, test_df = getAdData('cv')
            X_train_user = getUserData('cv')
            X_train = np.concatenate((X_train, X_train_user), axis=1)
        return X_train, Y_train, test_df
    elif train_type == 'val':
        if os.path.exists('tmp/tmp_tri_train_final_data.npy') and os.path.exists('tmp/tmp_tri_train_final_data.npy'):
            X_train = np.load('tmp/tmp_tri_train_final_data.npy')
            X_test = np.load('tmp/tmp_tri_train_final_data.npy')
            _, Y_train, _, Y_test, test_df = getAdData('val')

        else:
            X_train, Y_train, X_test, Y_test, test_df = getAdData('val')
            X_train_user, X_test_user = getUserData('val')
            X_train = np.concatenate((X_train, X_train_user), axis=1)
            X_test = np.concatenate((X_test, X_test_user), axis=1)
        return X_train, Y_train, X_test, Y_test, test_df
    elif train_type == 'test':
        if os.path.exists('tmp/tmp_train_final_data.npy'):
            X_train = np.load('tmp/tmp_train_final_data.npy')
            X_test = np.load('tmp/tmp_test_final_data.npy')
            _, Y_train, _, test_df = getAdData('test')
        else:
            X_train, Y_train, X_test, test_df = getAdData('test')
            X_train_user, X_test_user = getUserData('test')
            X_train = np.concatenate((X_train, X_train_user), axis=1)
            X_test = np.concatenate((X_test, X_test_user), axis=1)
        return X_train, Y_train, X_test, test_df
    else:
        raise Exception('Error while loading data')