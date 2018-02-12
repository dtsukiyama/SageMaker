import os
import sagemaker
import boto3, re
import tarfile

from sagemaker import get_execution_role
from dnn_classifier import train_input_fn
from dnn_classifier import estimator_fn
from dnn_classifier import serving_input_fn
from sagemaker.tensorflow.model import TensorFlowModel
from operator import itemgetter

sagemaker_session = sagemaker.Session()

role = get_execution_role()


def train(data_directory, num_steps=1000):
    """
    Args: 1. directory name where training data is stored
          2. Number of steps for training; default 1000
          
    Returns: 1. Trained canned estimator
    """
    classifier = estimator_fn(run_config = None, params = None)
    train_func = train_input_fn(data_directory, params = None)
    classifier.train(input_fn = train_func, steps = num_steps)
    # score
    score = classifier.evaluate(input_fn = train_func, steps = 100)
    print("model evaluation: {}".format(score))
    return classifier

def saveAndDeploy(model):
    """
    Args: 1. trained estimator
    
    Returns: 1. deployed model
    """
    exported_model = model.export_savedmodel(export_dir_base = 'export/Servo/', 
                                             serving_input_receiver_fn = serving_input_fn)
    print (exported_model)
    with tarfile.open('model.tar.gz', mode='w:gz') as archive:
        archive.add('export', recursive=True)
    print("model exported")
    sagemaker_session = sagemaker.Session()
    inputs = sagemaker_session.upload_data(path='model.tar.gz', key_prefix='model')
    sagemaker_model = TensorFlowModel(model_data = 's3://' + sagemaker_session.default_bucket() + '/model/model.tar.gz',
                                      role = role,
                                      entry_point = 'dnn_classifier.py')
    print("model saved")
    predictor = sagemaker_model.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
    return predictor

class serve(object):
    def __init__(self, model, encoder):
        """
        Args: predictor from deployed endpoint
        """
        self.model = model
        self.text_labels = encoder.classes_
    
    def predict(self, features):
        predictions = self.model.predict(features)
        return self.classPredictions(predictions)
    
    def delete(self):
        sagemaker.Session().delete_endpoint(self.model.endpoint)
        
    def classPredictions(self, predictions):
        predictions = predictions['result']['classifications'][0]['classes']
        predictions = sorted(predictions, key=itemgetter('score'), reverse=True)[:3]
        top3 = dict()
        for b in predictions:
            top3[self.text_labels[int(b['label'])]] = '{:.2%}'.format(b['score'])
        return top3


