import time


###########
# @Description: 文件路径保存工具
# @Param:
# @return:
# @Author: hancyran
# @Date: 2019-05-17 19:02
###########
from src.utils.path_args import args


def getModelPath(model):
    date = time.strftime('%m-%d %H:%M', time.localtime(time.time() + 3600 * 8))
    model_path = args.lgb_model_path + model + " " + date + ".model"
    return model_path


def getResultPath():
    date = time.strftime('%m-%d %H:%M', time.localtime(time.time() + 3600 * 8))
    model_path = args.result_path + date + ".csv"
    return model_path
