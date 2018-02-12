# Training and Deploying Deep Learning models on SageMaker


## Using canned estimators

Estimators, a high level TensorFlow API, simplifies your machine learning pipeline. Building an estimator allows you to train, evaluate, predict and export for serving. You can use one of TensorFlow's  pre-made classifier Estimators or you can buld your own. The first part of this tutorial addresses locally training a pre-made classifier and deploying an endpoint.

## Setup

This tutorial assumes you have SageMaker setup. You can clone this repo and upload the files to SageMaker or open up a terminal in SageMaker and clone the repo:

```
cd SageMaker
git clone https://github.com/dtsukiyama/SageMaker.git
cd SageMaker/notebooks

mkdir data
mkdir encoders
```

You will also need the dummy data we used for the classifier:

```
cd data
wget https://storage.googleapis.com/tensorflow-workshop-examples/stack-overflow-data.csv
cd ..
```

You can open the document-tagging.ipynb notebook to run the code.

## Data processing

We are using a dataset of Stack Overflow questions with tags. We need to tokenize the text features and encode the labels. Create training, testing, and hold out data sets.

```python
import pandas as pd
import numpy as np
from collections import Counter
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

def tokenizeText(data, num_words):
    """
    Bag of words tokenizer
    Args: training features and number of words
    Returns: bag of words matrix
    """
    tokenize = Tokenizer(num_words=num_words)
    tokenize.fit_on_texts(data)
    joblib.dump(tokenize, 'encoders/tokenize.pkl')
    X = tokenize.texts_to_matrix(data)
    return X

def encodeLabels(labels):
    encoder = LabelEncoder()
    encoder.fit(labels)
    y = encoder.transform(labels)
    num_classes = np.max(y) + 1
    print("num classes: {}".format(num_classes))
    joblib.dump(encoder, 'encoders/encoder.pkl')
    return y

data = pd.read_csv('data/stack-overflow-data.csv')

X = pd.DataFrame(tokenizeText(data['post'], 500))

sample = pd.concat([X, pd.DataFrame(encodeLabels(data['tags']))], axis=1)

train, test = train_test_split(sample, test_size=0.15)

test, holdout = train_test_split(test, test_size=0.1)

train.to_csv('data/post_train.csv',header=None,index=False)

test.to_csv('data/post_test.csv',header=None,index=False)

holdout.to_csv('data/post_holdout.csv', header=None,index=False)
```

## Training and deploying estimator

We import our libraries and define our role. We have some utility function in processing.py to handle loading our saved feature tokenizer and label encoder. 


```python
import os
import sagemaker
import boto3, re

from sagemaker import get_execution_role
from processing import myEncoder, myTokenizer
from sageBot import train, saveAndDeploy, serve

sagemaker_session = sagemaker.Session()
role = get_execution_role()
```

I have created a wrapper, sageBot.py to handle training, deployment, and serving. We import our estimator, dnn_classifier.py, to sageBot, it consists of four functions: estimator_fn, serving_input_fn, train_input_fn, and _generate_input_fn:

```python
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

INPUT_TENSOR_NAME = 'inputs'


def estimator_fn(run_config, params):
    """
    Args: config, params = None, None
    Returns: Deep neural net classifier pre-made estimator
    """
    feature_columns = [tf.feature_column.numeric_column(INPUT_TENSOR_NAME, shape=[500])]
    return tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                      hidden_units=[128, 256, 20],
                                      n_classes=20,
                                      config=run_config)

def serving_input_fn():
    feature_spec = {INPUT_TENSOR_NAME: tf.FixedLenFeature(dtype=tf.float32, shape=[500])}
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()

def train_input_fn(training_dir, params):
    """Returns input function that would feed the model during training"""
    return _generate_input_fn(training_dir, 'post_train.csv')


def _generate_input_fn(training_dir, training_filename, shuffle=False):
    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=os.path.join(training_dir, training_filename), target_dtype=np.int, features_dtype=np.float32)
    
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={INPUT_TENSOR_NAME:  np.array(training_set.data)}, 
        y=np.array(training_set.target),
        num_epochs=100,
        shuffle=shuffle
    )
    return input_fn

```

Our estimator in imported in sageBot for training and deployment.

```python
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
    Trains canned deep neural network estimator.
    
    Use: classifier = train('data')
    
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
    Takes trained classifier from train(), exports model to S3 bucket as tar.gz. Deploys model as endpoint.
    Returns model for use in serving.
    
    Use: predictor = saveAndDeploy(classifier)
    
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
        Takes exported model and saved encoder to serve enpoint predictions.
        
        Use: hosted = serve(predictor, encoder)
        
        hosted.predict(features)
        
        hosted.delete()
        
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


```

We can run the following to train and deploy, deploying takes several minutes:

```python
classifier = train('data')

predictor = saveAndDeploy(classifier)
```

We can serve predictions by doing the following:

```python
encoder = myEncoder()

hosted = serve(predictor, encoder)
```

Load some hold out data:

```python

df = pd.read_csv('data/post_holdout.csv')
sample = df.ix[100][:500].values
```

Predict:

```python
hosted.predict(sample)
```
```
{'.net': '0.44%', 'android': '0.62%', 'java': '98.84%'}
```

Check the prediction:

```python
test = int(df.ix[100][500])

encoder.classes_[test]
```
```
'java'
```

Delete the endpoint:

```python
hosted.delete()

```
