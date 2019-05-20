import pandas as pd


def getSample(act, pred):
    sample = sum(2 * abs(pred - act)/(pred + act)) / pred.size
    return sample


def getMonoScore(test_df, preds):
    test_sample_df = test_df[['广告id', '曝光广告出价bid']]

    test_sample_df['预测曝光量'] = preds
    test_sample_df.sort_values(by=["广告id", "曝光广告出价bid"], inplace=True)

    # 作为基准
    standard = test_sample_df.groupby(by='广告id').head(1)
    standard.rename(columns={'曝光广告出价bid': '基准出价', '预测曝光量': '基准曝光量'}, inplace=True)

    test_sample_df = pd.merge(test_sample_df, standard, how="left", left_on='广告id', right_on='广告id')

    def getScore(x):
        if x['基准曝光量'] == x['预测曝光量'] or x['基准出价'] == x['曝光广告出价bid']:
            return 1
        else:
            return ((x['基准曝光量'] - x['预测曝光量']) * (x['基准出价'] - x['曝光广告出价bid'])) / abs(
                (x['基准曝光量'] - x['预测曝光量']) * (x['基准出价'] - x['曝光广告出价bid']))

    test_sample_df['score'] = test_sample_df.apply(lambda x: getScore(x), axis=1)

    monoscore = test_sample_df.groupby(by='广告id')['score'].mean().mean()
    #     print("经过相关性计算成绩为："+str(monoscore))
    return monoscore

