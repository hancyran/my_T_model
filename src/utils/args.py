class args:
    root_path = '/jet/workspace/2019Tencent'

    log_path = root_path+ '/log'
    # data_path = '/data'
    train_data_path = root_path+  '/train2'
    test_data_path = root_path+'/test2'
    combine_feat_data_path = root_path+'/combine'
    tmp_data_path = root_path+'/tmp'

    feat2_list = ['广告id', '日期', '曝光广告素材尺寸', '广告行业id', '商品类型', '商品id', '广告账户id',
                  '创建日期', '创建星期', '投放时段']

    feat3_list = ['广告id', '日期', '曝光广告素材尺寸', '广告行业id', '商品类型', '商品id', '广告账户id',
                  '创建日期', '创建星期', '地域', '行为兴趣', '学历', '消费能力', '面向人群', '投放时段']
    ad_feat_dict = {'广告id': 0, '日期': 1, '曝光广告素材尺寸': 2, '广告行业id': 3,
                    '商品类型': 4, '商品id': 5, '广告账户id': 6, '曝光广告出价bid': 7,
                    '创建日期': 8, '创建星期': 9, '投放时段': range(10, 58)}
    ad_feat_en_dict = {'广告id': 'id', '日期': 'request_date', '曝光广告素材尺寸': 'size', '广告行业id': 'ad_factory_id',
                       '商品类型': 'product_type', '商品id': 'product_id', '广告账户id': 'acc_id', '曝光广告出价bid': 'bid',
                       '创建日期':'create_date', '创建星期': 'create_weekday', '投放时段': 'preriod'}
    user_feat_dict = {'年龄': range(0, 995), '性别': range(995, 1001), '地域': range(1001, 1001 + 4455),
                      '婚恋状况': range(5456, 5456 + 15), '工作状态': range(5471, 5471 + 6), '行为兴趣': range(5477, 5477 + 19711),
                      '设备': range(25188, 25188 + 6), '连接类型': range(25194, 25194 + 6), '学历': range(25200, 25200 + 8),
                      '消费能力': range(25208, 25208 + 5)}
    user_feat_en_dict = {'年龄': 'user_age', '性别': 'user_gender', '地域': 'user_area', '婚恋状况': 'user_status',
                         '工作状态': 'user_work', '行为兴趣': 'user_behavior', '设备': 'user_device',
                         '连接类型': 'user_connectionType', '学历': 'user_education', '消费能力': 'user_consuptionAbility'}


