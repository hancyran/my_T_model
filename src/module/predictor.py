import pandas as pd


def getPreds(model, X_test, test_df, pred_type='lr'):
    if pred_type == 'lr':
        test_sample_df = test_df[['广告id', '曝光广告出价bid']]
        #         test_sample_df.sort_values(by=["广告id","曝光广告出价bid"],inplace=True)

        standard = test_sample_df.groupby(by='广告id', sort=False).head(1)
        standard = standard.reset_index(drop=True)
        standard.rename(columns={'曝光广告出价bid': '基准bid'}, inplace=True)

        standard_index = standard.index
        preds = model.predict(X_test[standard_index])
        standard['基准预测值'] = preds

        test_sample_df = pd.merge(test_sample_df, standard, how="left", left_on='广告id', right_on='广告id')

        test_sample_df['preds'] = test_sample_df.apply(lambda x: x['基准预测值'] + 0.1 * (x['曝光广告出价bid'] - x['基准bid']),
                                                       axis=1)
        #         test_sample_df.sort_index(inplace=True)

        return test_sample_df['preds'].values
    elif pred_type == 'direct':
        # 直接用模型进行预测，与其他规则无关
        return model.predict(X_test)
    else:
        raise Exception('No such Predict Method')
