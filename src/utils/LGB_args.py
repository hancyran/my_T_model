class args:
    # hyperP
    # max_depth = [1, 3, 5, 7]
    max_depth = 7
    # learning_rate = [2, 1, 0.5, 0.1, 0.01]
    # learning_rate = [0.2, 0.05]
    learning_rate = 0.1
    # n_estimators = [100, 500, 1000, 1500, 2000, 3000, 5000, 7000, 10000]
    n_estimators = 1500
    num_leaves = [29, 31, 40, 50, 63, 80]
    reg_lambda = [0, 1, 3, 5, 7]
    feature_fraction = [0.5, 0.8, 0.9, 1.0]
    # model select
    boosting = ['gbdt', 'dart']
    objective = ['fair', 'regression_l1', 'huber', 'poisson']
    # standard config
    threads = 32
