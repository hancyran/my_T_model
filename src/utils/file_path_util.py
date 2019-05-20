import time


###########
# @Description: 文件路径保存工具
# @Param:
# @return:
# @Author: hancyran
# @Date: 2019-05-17 19:02
###########

def getModelPath(model):
    date = time.strftime('%m-%d %H:%M', time.localtime(time.time() + 3600 * 8))
    model_path = "model/" + model + " " + date + ".model"
    return model_path


def getResultPath():
    date = time.strftime('%m-%d %H:%M', time.localtime(time.time() + 3600 * 8))
    model_path = "result/" + date + ".result"
    return model_path
