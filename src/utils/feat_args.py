class fargs:
    feat2_list = ['广告id', '日期', '曝光广告素材尺寸', '广告行业id', '商品类型', '商品id', '广告账户id',
                  '创建日期', '创建星期', '投放时段']

    feat3_list = ['广告id', '日期', '曝光广告素材尺寸', '广告行业id', '商品类型', '商品id', '广告账户id',
                  '创建日期', '创建星期', '地域', '行为兴趣', '学历', '消费能力', '面向人群', '投放时段']

    ad_feat_en_dict = {'广告id': 'id', '日期': 'request_date', '曝光广告素材尺寸': 'size', '广告行业id': 'ad_factory_id',
                       '商品类型': 'product_type', '商品id': 'product_id', '广告账户id': 'acc_id', '曝光广告出价bid': 'bid',
                       '创建日期': 'create_date', '创建星期': 'create_weekday', '投放时段': 'period'}

    user_feat_en_dict = {'年龄': 'user_age', '性别': 'user_gender', '地域': 'user_area', '婚恋状况': 'user_status',
                         '工作状态': 'user_work', '行为兴趣': 'user_behavior', '设备': 'user_device',
                         '连接类型': 'user_connectionType', '学历': 'user_education', '消费能力': 'user_consuptionAbility'}

    ad_feat_dict = {'id': list(range(0, 31154)), 'create_weekday': list(range(31154, 31154 + 7)),
                    'ad_factory_id': list(range(31161, 31161 + 195)),
                    'product_type_id': list(range(31356, 31356 + 10)), 'product_id': list(range(31366, 31366 + 12643)),
                    'acc_id': list(range(44009, 44009 + 9491)), 'period': list(range(53500, 53500 + 48)),
                    'request_date_id': [53548],
                    'create_date_id': [53549], 'bid': [53550], 'size': [53551]}
    ad_feat = list(ad_feat_dict.keys())

    user_feat_dict = {'user_age': list(range(53553, 53553 + 995)), 'user_gender': list(range(54548, 54548 + 6)),
                      'user_area': list(range(54554, 54554 + 4419)),
                      'user_status': list(range(58973, 58973 + 15)), 'user_work': list(range(58988, 58988 + 6)),
                      'user_behavior': list(range(58994, 58994 + 19752)),
                      'user_device': list(range(78746, 78746 + 10)),
                      'user_connectionType': list(range(78756, 78756 + 6)),
                      'user_education': list(range(78762, 78762 + 8)),
                      'user_consuptionAbility': list(range(78770, 78770 + 5))}
    user_feat = list(user_feat_dict.keys())
    all_feat = ad_feat + user_feat

    missing_feat = ['ad_factory_id', 'product_id']

    # dfm feats
    onehot_feats = ['id', 'create_weekday', 'ad_factory_id', 'product_type_id', 'product_id', 'acc_id']
    numeric_feats = ['request_date_id', 'create_date_id', 'bid', 'size']
    sequence_feats = ['period'] + list(user_feat_dict.keys())

    max_len_for_feats = {'period': 48, 'user_age': 1000, 'user_gender': 6, 'user_area': 4419, 'user_status': 15,
                         'user_work': 6, 'user_behavior': 19752,
                         'user_device': 10, 'user_connectionType': 6, 'user_education': 8, 'user_consuptionAbility': 5}


