import os
import shutil

import numpy as np
import pandas as pd
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from src.utils.feat_args import fargs
from src.utils.path_args import pargs


# def getDfmData(train_type):
#     if train_type == 'cv':
#         if os.path.exists(pargs.tmp_data_path + '/tmp_train_final_data.npy'):
#             X_train = np.load(pargs.tmp_data_path + '/tmp_train_final_data.npy')
#             Y_train = np.load(pargs.train_data_path + '/train_arr_label.npy')
#
#             shutil.copy(pargs.train_data_path + '/train_origin_final.h5', './')
#             test_df = pd.read_hdf('train_origin_final.h5')
#             # _, Y_train, test_df = getAdData('cv')
#         else:
#             X_train, Y_train, test_df = getAdData('cv')
#             X_train_user = getUserData('cv')
#             X_train = np.concatenate((X_train, X_train_user), axis=1)
#         return X_train, Y_train, test_df
#
#     elif train_type == 'val':
#         if os.path.exists(pargs.tmp_data_path + '/tmp_tri_train_final_data.npy') \
#                 and os.path.exists(pargs.tmp_data_path + '/tmp_tri_train_final_data.npy'):
#             X_train = np.load(pargs.tmp_data_path + '/tmp_tri_train_final_data.npy')
#             X_test = np.load(pargs.tmp_data_path + '/tmp_tri_train_final_data.npy')
#             label = np.load(pargs.train_data_path + '/train_arr_label.npy')
#
#             shutil.copy(pargs.train_data_path + '/train_origin_final.h5', './')
#             test_df = pd.read_hdf('train_origin_final.h5')
#             i = test_df.loc[test_df.日期 == 319].index
#             m = test_df.loc[test_df.日期 != 319].index
#
#             Y_train = label[m]
#             Y_test = label[i]
#             # _, Y_train, _, Y_test, test_df = getAdData('val')
#         else:
#             X_train, Y_train, X_test, Y_test, test_df = getAdData('val')
#             X_train_user, X_test_user = getUserData('val')
#             X_train = np.concatenate((X_train, X_train_user), axis=1)
#             X_test = np.concatenate((X_test, X_test_user), axis=1)
#         return X_train, Y_train, X_test, Y_test, test_df
#
#     elif train_type == 'test':
#         if os.path.exists(pargs.tmp_data_path + '/tmp_train_final_data.npy'):
#             X_train = np.load(pargs.tmp_data_path + '/tmp_train_final_data.npy')
#             X_test = np.load(pargs.tmp_data_path + '/tmp_test_final_data.npy')
#
#             Y_train = np.load(pargs.train_data_path + '/train_arr_label.npy')
#
#             shutil.copy(pargs.test_data_path + '/test_origin_final.h5', './')
#             test_df = pd.read_hdf('test_origin_final.h5')
#             # _, Y_train, _, test_df = getAdData('test')
#         else:
#             X_train, Y_train, X_test, test_df = getAdData('test')
#             X_train_user, X_test_user = getUserData('test')
#             X_train = np.concatenate((X_train, X_train_user), axis=1)
#             X_test = np.concatenate((X_test, X_test_user), axis=1)
#         return X_train, Y_train, X_test, test_df
#
#     else:
#         raise Exception('Error while loading data')
def getDfmData(train_type):
    def convert(x):
        return np.where(x)[0].tolist()
        # return [np.where(y) if np.where(y)[0].tolist() == [] else 0 for y in x]

    def arr_to_list(x):
        if x.tolist():
            return x.tolist()[0]
        else:
            return 0

    if train_type == 'cv':
        dic = dict()
        Y_train = np.load(pargs.train_data_path + '/train_arr_label.npy')
        test_df = pd.read_hdf(pargs.train_data_path + '/train_origin_final.h5')

        dic['id'] = np.nonzero(np.load(pargs.train_data_path + '/train_arr_id.npy'))[1] + 1

        for n, i in enumerate(fargs.all_feat):
            if n == 0:
                pass
            elif n < 6 and i not in fargs.missing_feat:
                # X_train = np.concatenate((X_train, np.load(pargs.train_data_path + '/train_arr_%s.npy' % i)), axis=1)
                dic[i] = np.nonzero(np.load(pargs.train_data_path + '/train_arr_%s.npy' % i))[1] + 1
            elif i in fargs.missing_feat:
                dic[i] = np.array([arr_to_list(np.where(y)[0]) for y in
                                   np.load(pargs.train_data_path + '/train_arr_%s.npy' % i)]) + 1

            elif 7 <= n <= 10:
                # X_train = np.concatenate((X_train, np.load(pargs.train_data_path + '/train_arr_%s_scaled.npy' % i)),
                # axis=1)
                dic[i] = np.load(pargs.train_data_path + '/train_arr_%s_scaled.npy' % i)
            elif i in fargs.sequence_feats:
                X_train = np.load(pargs.train_data_path + '/train_arr_%s.npy' % i)
                typ = list(map(convert, (X_train == 1)))
                typ = [list(x+1 for x in y) for y in typ]
                dic[i] = pad_sequences(typ, maxlen=fargs.max_len_for_feats.get(i), padding='post', )

        return dic, Y_train, test_df
    elif train_type == 'val':
        tri_dic = dict()
        val_dic = dict()

        Y_train = np.load(pargs.train_data_path + '/train_arr_label.npy')
        test_df = pd.read_hdf(pargs.train_data_path + '/train_origin_final.h5')

        # tri
        m = test_df.loc[test_df.日期 != 319].index
        # val
        k = test_df.loc[test_df.日期 == 319].index

        Y_tri_train = Y_train[m]
        Y_val_train = Y_train[k]

        tri_dic['id'] = np.nonzero(np.load(pargs.train_data_path + '/train_arr_id.npy'))[1]

        for n, i in enumerate(fargs.all_feat):
            if n == 0:
                pass
            elif n < 6:
                # X_train = np.concatenate((X_train, np.load(pargs.train_data_path + '/train_arr_%s.npy' % i)), axis=1)
                arr = np.nonzero(np.load(pargs.train_data_path + '/train_arr_%s.npy' % i))[1]
                tri_dic[i] = arr[m]
                val_dic[i] = arr[k]
            elif 7 <= n <= 10:
                # X_train = np.concatenate((X_train, np.load(pargs.train_data_path + '/train_arr_%s_scaled.npy' % i)),
                # axis=1)
                arr = np.load(pargs.train_data_path + '/train_arr_%s_scaled.npy' % i)
                tri_dic[i] = arr[m]
                val_dic[i] = arr[k]
            else:
                X_train = np.load(pargs.train_data_path + '/train_arr_period.npy')
                typ = list(map(convert, (X_train == 1)))
                arr = pad_sequences(typ, maxlen=fargs.max_len_for_feats.get(i), padding='post', )
                tri_dic[i] = arr[m]
                val_dic[i] = arr[k]
        return tri_dic, Y_tri_train, val_dic, Y_val_train, test_df

    elif train_type == 'test':
        train_dic = dict()
        test_dic = dict()

        Y_train = np.load(pargs.train_data_path + '/train_arr_label.npy')
        test_df = pd.read_hdf(pargs.test_data_path + '/test_origin_final.h5')

        train_dic['id'] = np.nonzero(np.load(pargs.train_data_path + '/train_arr_id.npy'))[1]

        for n, i in enumerate(fargs.all_feat):
            if n == 0:
                pass
            elif n < 6:
                # X_train = np.concatenate((X_train, np.load(pargs.train_data_path + '/train_arr_%s.npy' % i)), axis=1)
                train_dic[i] = np.nonzero(np.load(pargs.train_data_path + '/train_arr_%s.npy' % i))[1]
            elif 7 <= n <= 10:
                # X_train = np.concatenate((X_train, np.load(pargs.train_data_path + '/train_arr_%s_scaled.npy' % i)),
                # axis=1)
                train_dic[i] = np.load(pargs.train_data_path + '/train_arr_%s_scaled.npy' % i)
            else:
                X_train = np.load(pargs.train_data_path + '/train_arr_period.npy')
                typ = list(map(convert, (X_train == 1)))
                train_dic[i] = pad_sequences(typ, maxlen=fargs.max_len_for_feats.get(i), padding='post', )

        test_dic['id'] = np.nonzero(np.load(pargs.test_data_path + '/test_arr_id.npy'))[1]

        for n, i in enumerate(fargs.all_feat):
            if n == 0:
                pass
            elif n < 6:
                # X_train = np.concatenate((X_train, np.load(pargs.train_data_path + '/train_arr_%s.npy' % i)), axis=1)
                test_dic[i] = np.nonzero(np.load(pargs.test_data_path + '/test_arr_%s.npy' % i))[1]
            elif 7 <= n <= 10:
                # X_train = np.concatenate((X_train, np.load(pargs.train_data_path + '/train_arr_%s_scaled.npy' % i)),
                # axis=1)
                test_dic[i] = np.load(pargs.test_data_path + '/test_arr_%s_scaled.npy' % i)
            else:
                X_test = np.load(pargs.test_data_path + '/test_arr_period.npy')
                typ = list(map(convert, (X_test == 1)))
                test_dic[i] = pad_sequences(typ, maxlen=fargs.max_len_for_feats.get(i), padding='post', )

        return train_dic, Y_train, test_dic, test_df
