import pandas as pd
import numpy as np

from src.utils.file_path_util import getResultPath


def create_submit(final_preds):
    out_predict_result = np.around(final_preds, decimals=4)

    predict = pd.DataFrame(out_predict_result).reset_index()
    predict['index'] = predict['index'].apply(lambda x: x + 1)

    predict.to_csv(getResultPath(), header=0, index=0)
    predict.to_csv('submission.csv', header=0, index=0)