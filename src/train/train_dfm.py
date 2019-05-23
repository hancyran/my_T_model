# val : test with validation set()
# test: predict with original test set
# cv: use cross validation for model testing
import time

import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold

from src.eval.metric import getSample, getMonoScore
from src.feat.deepfm_data_getter import getDfmData
from src.models.deepfm import DeepFM
from src.module.predictor import getPreds
from src.utils.feat_args import fargs
from src.utils.file_path_util import getModelPath
from src.utils.misc_util import checkPos

# val : test with validation set()
# test: predict with original test set
# cv: use cross validation for model testing
from src.utils.submission_creator import create_submit
from src.utils.dfm_utils import SingleFeat, VarLenFeat


def trainDFM(train_type='cv'):
    # read data
    print('Loading data...')
    if train_type == 'cv':
        X_train, Y_train, test_df = getDfmData('cv')
    elif train_type == 'val':
        X_train, Y_train, X_test, Y_test, test_df = getDfmData('val')
    elif train_type == 'test':
        X_train, Y_train, X_test, test_df = getDfmData('test')

    feats = fargs.all_feat
    ad_feat_dict = fargs.ad_feat_dict
    # feats
    onehot_feats = fargs.onehot_feats
    numeric_feats = fargs.numeric_feats
    user_feat_dict = fargs.user_feat_dict
    max_len_dict = fargs.max_len_for_feats

    # prepare for feat field
    sparse_feat_list = []
    sequence_feat_list = []
    dense_feat_list = []
    for i in feats:
        if i in onehot_feats:
            sparse_feat_list += [SingleFeat(i, len(ad_feat_dict.get(i)))]
        elif i in numeric_feats:
            dense_feat_list += [SingleFeat(i, 0)]
        elif i == 'period':
            sequence_feat_list += [VarLenFeat(i, len(ad_feat_dict.get(i)), max_len_dict.get(i), 'mean')]
        elif i in user_feat_dict:
            sequence_feat_list += [VarLenFeat(i, len(user_feat_dict.get(i)), max_len_dict.get(i), 'mean')]

    # create regressor
    model = DeepFM({"sparse": sparse_feat_list, "dense": dense_feat_list, "sequence": sequence_feat_list},
                   task='regression')

    model.compile("adam", "mse", metrics=['mse'])

    ######### start cv training #############

    # cross validation

    if train_type == 'cv':
        print("Start CV Training...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
        #
        sample_list = []
        mono_score_list = []

        sparse_input = [X_train.get(feat) for feat in X_train.keys() if feat in fargs.onehot_feats]
        dense_input = [X_train.get(feat) for feat in X_train.keys() if feat in fargs.numeric_feats]
        sequence_input = [X_train.get(feat) for feat in X_train.keys() if feat in fargs.sequence_feats]
        model_input = sparse_input + dense_input + sequence_input

        folds = list(skf.split(np.zeros(178541), np.zeros(178541)))
        # X_train = list(X_train.values())
        for i, (train, test) in enumerate(folds):
            print("Fold: ", i)
            start = time.time()

            train_data = [x[train] for x in model_input]
            test_data = [x[test] for x in model_input]
            # train model
            # model.fit(X_train[train], Y_train[train], eval_metric='l1', categorical_feature=[0, 3, 4, 5, 6, 9],
            #           #                       early_stopping_rounds=100,
            #           eval_set=[(X_train[train], Y_train[train]), (X_train[test], Y_train[test])]
            #           )
            model.fit(train_data, Y_train[train], batch_size=256, epochs=10, verbose=2)
            # predict
            #             preds = model.predict(X_train[test])
            preds = getPreds(model, test_data, test_df.iloc[test], pred_type='dfm')
            preds = np.array([checkPos(x) for x in preds])

            end = time.time()
            # output the cost time
            print("The fold cost %f mins" % ((int(end) - int(start)) / 60))

            # eval model with specific metric
            sample = getSample(preds, Y_train[test])
            sample_list += [sample]
            mono_score = getMonoScore(test_df.iloc[test], preds)
            mono_score_list += [mono_score]
            # print each split
            print("Sample: %f \n" % sample)
            print("Score: %f" % mono_score)
            print("Total Score: %f" % (0.4 * (1 - sample / 2) + 0.6 * (mono_score + 1) / 2))
        print("End CV Training...")
        # print eval result
        final_sample = sum(sample_list) / 10
        final_mono_score = sum(mono_score_list) / 10
        print("Final Sample: %f " % final_sample)
        print("Final MonoScore: %f " % final_mono_score)
        print("Final TotalScore: %f" % (0.4 * (1 - final_sample / 2) + 0.6 * (final_mono_score + 1) / 2))
        print("Expect TotalScore: %f" % (0.4 * (1 - final_sample / 2) + 0.6 * (1 + 1) / 2))
        print("Saving Model " + getModelPath('lightGBM-cv'))
        #     model.save_model(getModelPath('lightGBM-cv'))
        joblib.dump(model, getModelPath('lightGBM-cv'))
    ######### end cv training #############

    ######### start val training #############

    # cross validation
    if train_type == 'val':
        print("Start Val Training...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
        #
        sample_list = []
        mono_score_list = []
        final_preds = np.zeros_like(Y_test)
        for i, (train, test) in enumerate(skf.split(X_train, Y_train)):
            print("Fold: ", i)
            start = time.time()
            # train model
            model.fit(X_train[train], Y_train[train], eval_metric='l1', categorical_feature=[0, 3, 4, 5, 6, 9],
                      #                       early_stopping_rounds=100,
                      eval_set=[(X_train[train], Y_train[train]), (X_train[test], Y_train[test])]
                      )
            # predict
            preds = getPreds(model, X_test, pred_type='dfm')
            preds = np.array([checkPos(x) for x in preds])
            final_preds = preds * 0.1 + final_preds
            end = time.time()
            # output the cost time
            print("The fold cost %f mins" % ((int(end) - int(start)) / 60))

            # eval model with specific metric
            sample = getSample(preds, Y_test)
            sample_list += [sample]
            mono_score = getMonoScore(test_df.loc[test_df.日期 == 319], preds)
            mono_score_list += [mono_score]
            # print each split
            print("Sample: %f \n" % sample)
            print("Score: %f" % mono_score)
            print("Total Score: %f" % (0.4 * (1 - sample / 2) + 0.6 * (mono_score + 1) / 2))
        print("End Val Training...")
        # print eval result
        final_sample = getSample(final_preds, Y_test)
        final_mono_score = getMonoScore(test_df.loc[test_df.日期 == 319], final_preds)
        print("Final Sample: %f " % final_sample)
        print("Final MonoScore: %f " % final_mono_score)
        print("Final TotalScore: %f" % (0.4 * (1 - final_sample / 2) + 0.6 * (final_mono_score + 1) / 2))
        print("Expect TotalScore: %f" % (0.4 * (1 - final_sample / 2) + 0.6 * (1 + 1) / 2))
        print("Saving Model " + getModelPath('lightGBM-val'))
        #     model.save_model(getModelPath('lightGBM-val'))
        joblib.dump(model, getModelPath('lightGBM-val'))
    ######### end val training #############

    ######### start testing dataset #############

    # training whole dataset
    if train_type == 'test':
        print("Start Test Training...")
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2019)
        #
        sample_list = []
        mono_score_list = []
        final_preds = np.zeros(X_test.shape[0])
        for i, (train, test) in enumerate(skf.split(X_train, Y_train)):
            print("Fold: ", i)
            start = time.time()
            # train model
            model.fit(X_train[train], Y_train[train], eval_metric='l1', categorical_feature=[0, 3, 4, 5, 6, 9],
                      #                       early_stopping_rounds=500,
                      eval_set=[(X_train[train], Y_train[train]), (X_train[test], Y_train[test])]
                      )
            # predict
            preds = getPreds(model, X_test, test_df)
            final_preds = final_preds + preds * 0.1
            end = time.time()
            # output the cost time
            print("The fold cost %f mins" % ((int(end) - int(start)) / 60))
        print("End Test Training...")
        # save model
        print("Saving Model " + getModelPath('lightGBM-test'))
        #         model.save_model(getModelPath('lightGBM-test'))
        joblib.dump(model, getModelPath('lightGBM-test'))
        ######### end testing dataset #############

        ############### export preds as csv ###################
        create_submit(final_preds)

    ############# plot if needed ######################
    #     LGBplot(plot_importance, plot_metric)

    # if plot_importance:
    #     print('特征重要性排序...')
    #     print(model.feature_importances_)
    #     ax = lgb.plot_importance(model, max_num_features=100, figsize=(15, 30))  # max_features表示最多展示出前10个重要性特征，可以自行设置
    #     plt.show()
    # if plot_metric:
    #     print('训练结果图像...')
    #     ax = lgb.plot_metric(model, metric='l1')  # metric的值与之前的params里面的值对应
    #     plt.show()

    print(model.get_params())
    print(time.strftime('%m-%d %H:%M', time.localtime(time.time() + 3600 * 8)))

    print(model.get_params())
    print(time.strftime('%m-%d %H:%M', time.localtime(time.time() + 3600 * 8)))
