import xgboost as xgb


def createXGB(max_depth=5, learning_rate=0.1, n_estimators=100,  reg_alpha=0, reg_lambda=0):
    # create regressor
    model = xgb.XGBRegressor(max_depth=max_depth,
                             learning_rate=learning_rate,
                             n_estimators=n_estimators,
                             min_child_weight=1,
                             gamma=0,
                             subsample=0.8,
                             colsample_bytree=1,
                             reg_alpha=reg_alpha,
                             reg_lambda=reg_lambda,
                             scale_pos_weight=1,
                             seed=1024,
                             n_jobs=48,
                             objective='reg:linear', booster='gbtree', verbosity=1, silent=True,
                             max_delta_step=0, importance_type='gain', eval_metric='mae',
                             colsample_bylevel=1, colsample_bynode=1,
                             base_score=0.5, random_state=0, missing=None)
    return model
