from src.models.LGB import createLGB
from src.models.XGB import createXGB
from src.models.deepfm import DeepFM
from src.models.xdeepfm import xDeepFM


def createModel(type='LGB'):
    if type == 'LGB':
        return createLGB()
    elif type == 'XGB':
        return createXGB()
    elif type == 'DeepFM':
        return DeepFM()
    # elif type == 'DIEN':
    #     return createDIEN()
    elif type == 'XDeepFM':
        return xDeepFM()
    else:
        raise Exception('No such model')