import os
import gc

import numpy as np
import pandas as pd


def getUserData(train_type):
    if train_type == 'cv':
        #### train dataset
        if os.path.exists('tmp/tmp_train_data_user.npy'):
            X_train = np.load('tmp/tmp_train_data_user.npy')
        else:
            X_train = np.load('train2/train_arr_user_age.npy')
            X_train = np.concatenate((X_train, np.load('train2/train_arr_user_gender.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_user_area.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_user_status.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_user_work.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_user_behavior.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_user_device.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_user_connectionType.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_user_education.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_user_consuptionAbility.npy')), axis=1)

            # save as tmp file
            np.save('tmp/tmp_train_data_user.npy', X_train)

        return X_train

    elif train_type == 'val':
        # load tmp file if exists
        if os.path.exists('tmp/tmp_tri_train_data_user.npy') and os.path.exists('tmp/tmp_val_train_data_user.npy'):

            X_tri_data = np.load('tmp/tmp_tri_train_data_user.npy')
            X_val_data = np.load('tmp/tmp_val_train_data_user.npy')



        else:
            #### original train dataset
            if os.path.exists('tmp/tmp_train_data_user.npy'):
                X_train = np.load('tmp/tmp_train_data_user.npy')
            else:
                X_train = np.load('train2/train_arr_user_age.npy')
                X_train = np.concatenate((X_train, np.load('train2/train_arr_user_gender.npy')), axis=1)
                X_train = np.concatenate((X_train, np.load('train2/train_arr_user_area.npy')), axis=1)
                X_train = np.concatenate((X_train, np.load('train2/train_arr_user_status.npy')), axis=1)
                X_train = np.concatenate((X_train, np.load('train2/train_arr_user_work.npy')), axis=1)
                X_train = np.concatenate((X_train, np.load('train2/train_arr_user_behavior.npy')), axis=1)
                X_train = np.concatenate((X_train, np.load('train2/train_arr_user_device.npy')), axis=1)
                X_train = np.concatenate((X_train, np.load('train2/train_arr_user_connectionType.npy')), axis=1)
                X_train = np.concatenate((X_train, np.load('train2/train_arr_user_education.npy')), axis=1)
                X_train = np.concatenate((X_train, np.load('train2/train_arr_user_consuptionAbility.npy')), axis=1)

                # save as tmp file
                np.save('tmp/tmp_train_data_user.npy', X_train)

            # train_df
            test_df = pd.read_hdf('train/train_origin_final.h5')

            # pepare for dataset
            i = test_df.loc[test_df.日期 == 319].index
            m = test_df.loc[test_df.日期 != 319].index
            ### train dataset
            X_tri_data = X_train[m]
            np.save('tmp/tmp_tri_train_data_user.npy', X_tri_data)

            ### test dataset
            X_val_data = X_train[i]
            np.save('tmp/tmp_val_train_data_user.npy', X_val_data)

            ## collect
            del X_train
            gc.collect()

        return X_tri_data, X_val_data

    elif train_type == 'test':
        #### original rain dataset
        if os.path.exists('tmp/tmp_train_data_user.npy'):
            X_train = np.load('tmp/tmp_train_data_user.npy')
        else:
            X_train = np.load('train2/train_arr_user_age.npy')
            X_train = np.concatenate((X_train, np.load('train2/train_arr_user_gender.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_user_area.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_user_status.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_user_work.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_user_behavior.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_user_device.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_user_connectionType.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_user_education.npy')), axis=1)
            X_train = np.concatenate((X_train, np.load('train2/train_arr_user_consuptionAbility.npy')), axis=1)

            # save as tmp file
            np.save('tmp/tmp_train_data_user.npy', X_train)

        ### test dataset
        if os.path.exists('tmp/tmp_test_data_user.npy'):
            X_test = np.load('tmp/tmp_test_data_user.npy')
        else:
            X_test = np.load('test2/test_arr_user_age.npy')
            X_test = np.concatenate((X_test, np.load('test2/test_arr_user_gender.npy')), axis=1)
            X_test = np.concatenate((X_test, np.load('test2/test_arr_user_area.npy')), axis=1)
            X_test = np.concatenate((X_test, np.load('test2/test_arr_user_status.npy')), axis=1)
            X_test = np.concatenate((X_test, np.load('test2/test_arr_user_work.npy')), axis=1)
            X_test = np.concatenate((X_test, np.load('test2/test_arr_user_behavior.npy')), axis=1)
            X_test = np.concatenate((X_test, np.load('test2/test_arr_user_device.npy')), axis=1)
            X_test = np.concatenate((X_test, np.load('test2/test_arr_user_connectionType.npy')), axis=1)
            X_test = np.concatenate((X_test, np.load('test2/test_arr_user_education.npy')), axis=1)
            X_test = np.concatenate((X_test, np.load('test2/test_arr_user_consuptionAbility.npy')), axis=1)
            # save as tmp file
            np.save('tmp/tmp_test_data_user.npy', X_test)
        ###

        return X_train, X_test

    else:
        raise Exception('No Such Train Type')
