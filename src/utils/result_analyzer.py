import pandas as pd

from src.eval.metric import getMonoScore


def testSubmission(preds):
    preds = pd.read_csv('submission.csv', names=['id', 'quantity'], usecols=['quantity'])
    test_df = pd.read_hdf('test_data3.h5')
    ms = getMonoScore(test_df, preds)
    print('The MonoScore is %f' % ms)
    return ms

def getSampleFromTotalScore(preds, totalScore):
    ms = testSubmission(preds)
    sample = 3.5 + 1.5*0.3 - totalScore
    print('The Sample is %f' % sample)
    return sample