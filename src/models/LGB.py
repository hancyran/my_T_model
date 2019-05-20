import lightgbm as lgb


def createLGB(max_depth=3, learning_rate=1, n_estimators=6000, reg_alpha=0, reg_lambda=0):
    model = lgb.LGBMRegressor(max_depth=max_depth,
                              learning_rate=learning_rate,
                              #                               num_leaves=29,
                              n_estimators=n_estimators,
                              subsample=0.8,
                              colsample_bytree=0.7,
                              #                               subsample_for_bin=50000,
                              min_child_weight=1,
                              reg_alpha=reg_alpha,
                              reg_lambda=reg_lambda,
                              gamma=0,
                              scale_pos_weight=1,
                              #                               min_split_gain=0,
                              #                               max_bin=425,
                              #                               subsample_freq=1,
                              seed=1024,
                              boosting_type='gbdt',
                              #                               boosting_type='dart',
                              objective='regression',
                              nthread=48, silent=True)

    return model
