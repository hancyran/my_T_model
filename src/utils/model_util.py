import time
from keras.models import model_from_yaml
from keras.models import load_model

from src.utils.file_path_util import getModelPath


def saveModel(model, modelName='', file_type='h5'):
    if file_type == 'h5':
        model.save(getModelPath(modelName) + '.h5')

    elif file_type == 'yaml':
        with open(getModelPath(modelName) + '.yaml', 'w') as f:
            f.write(model.to_yaml())


def loadModel(file=None, type='h5'):
    if file:
        if type == 'h5':
            model = load_model(file)
            return model
        elif type == 'yaml':
            with open(file) as f:
                yml_str = f.read()
                model = model_from_yaml(yml_str)
                return model
    else:
        raise Exception('No Input FilePath')
