# 将预测值为负值的转化为1
def checkPos(x):
    if x < 0:
        return 1
    else:
        return x
