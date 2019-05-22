# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,wcshen1994@163.com

"""

from collections import namedtuple




try:
    from packaging.version import parse
except ImportError:
    from pip._vendor.packaging.version import parse


class SingleFeat(namedtuple('SingleFeat', ['name', 'dimension', 'hash_flag', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension, hash_flag=False, dtype="float32"):
        return super(SingleFeat, cls).__new__(cls, name, dimension, hash_flag, dtype)


class VarLenFeat(namedtuple('VarLenFeat', ['name', 'dimension', 'maxlen', 'combiner', 'hash_flag', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension, maxlen, combiner="mean", hash_flag=False, dtype="float32"):
        return super(VarLenFeat, cls).__new__(cls, name, dimension, maxlen, combiner, hash_flag, dtype)





def check_feature_config_dict(feature_dim_dict):
    if not isinstance(feature_dim_dict, dict):
        raise ValueError(
            "feature_dim_dict must be a dict like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4',"
            "'field_5']}") 
    if "sparse" not in feature_dim_dict:
        feature_dim_dict['sparse'] = []
    if "dense" not in feature_dim_dict:
        feature_dim_dict['dense'] = []
    if "sequence" not in feature_dim_dict:
        feature_dim_dict["sequence"] = []  # TODO:check if it's ok

    if not isinstance(feature_dim_dict["sparse"], list):
        raise ValueError("feature_dim_dict['sparse'] must be a list,cur is", type(
            feature_dim_dict['sparse']))

    if not isinstance(feature_dim_dict["dense"], list):
        raise ValueError("feature_dim_dict['dense'] must be a list,cur is", type(
            feature_dim_dict['dense']))
