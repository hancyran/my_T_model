# TENCENT AD ALGORITHM COMPETITION:  

## 特征工程：

* feat3：(忽略掉bid)
'广告id', '日期', '曝光广告素材尺寸', '广告行业id', '商品类型', '商品id', '广告账户id', 
'创建日期', '创建星期', '地域长度', '行为兴趣长度', '学历长度', '消费能力长度', '面向人群长度', '投放时段长度'

## 模型选用：

* LGB:
参数：
max_depth=max_depth,<br>
learning_rate=learning_rate,<br>
num_leaves=64, <br>
n_estimators=n_estimators, <br>
subsample=0.8, <br>
colsample_bytree=0.7,<br>
subsample_for_bin=50000, <br>
min_child_weight=1, <br>
reg_alpha=0,<br>
reg_lambda=5,<br>
gamma=0,<br>
scale_pos_weight=1,<br>
min_split_gain=0,<br>
max_bin=425, <br>
subsample_freq=1,<br>
seed=2019, <br>
boosting_type='gbdt',<br>
boosting_type='dart',<br>
objective='huber'<br>

* XGB:
参数：

