class args:

    max_depth = [1,3,5,7]
    learning_rate = [2,1,0.5,0.1,0.01]
    n_estimators = [100, 500, 1000, 3000, 5000, 7000, 10000]
    reg_lambda = [0, 1, 3, 5, 7]
    feature_fraction = [0.5, 0.8, 0.9, 1.0]
    objective = ['huber', 'regression', 'regression_l2', 'regression_l1', 'fair', 'poisson', 'mape']
