# Training and Deploying Machine Learning models on SageMaker


# Two types of estimators

- Pre-made estimators
- Custom estimators
  
  
# What is SageMaker?


Amazon states:

>Amazon SageMaker is a fully-managed service that enables developers and data scientists to quickly and easily build, train, and deploy machine learning models at any scale. Amazon SageMaker removes all the barriers that typically slow down developers who want to use machine learning.


If you come from a software development background, a data science and machine learning workflow may be unfamiliar to you. In general the workflow consists of the following:

1. Data exploration and feature engineering

2. Model building and training

3. Model evaluation and optimization

4. Model deployment

5. Iterate

Often this process takes place in [Jupyter notebooks](http://jupyter.org/). Jupyter notebooks allow you to share code, visualizations, and narrative text; which helps others understand the steps taken when modeling a task.



## Pre-made estimators 

Estimators, a high level TensorFlow API, simplifies your machine learning pipeline. Building an estimator allows you to train, evaluate, predict and export for serving. You can use one of TensorFlow's  pre-made classifier Estimators or you can buld your own. The first part of this tutorial addresses locally training a pre-made classifier and deploying an endpoint. The second part addresses custom estimators and training on SageMaker's infrastructure.

## Setup

This tutorial assumes you have SageMaker setup. You can clone this repo and upload the files to SageMaker or open up a terminal in SageMaker and try clone the repo* (I have not tried this, and it does not seem as if SageMaker has a way to version control yet).


There are two notebooks: ```document-tagging``` and ```cutom-model```.


We also need dummy data. in SageMaker, after you launch you Jupyter notebook instance you can open up a terminal. Download the testing data:


```
cd SageMaker

cd data-sagemaker-dev/test-models/data

wget https://storage.googleapis.com/tensorflow-workshop-examples/stack-overflow-data.csv

cd ..
```



## Data processing

We are using a dataset of Stack Overflow questions with tags. We need to tokenize the text features and encode the labels. Create training, testing, and hold out datasets.

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

We import our libraries and define our role. We have some utility functions in processing.py to handle loading our saved feature tokenizer and label encoder. 


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

I have created a wrapper, sageBot.py to handle training, deployment, and serving. We import our estimator, dnn_classifier.py, to sageBot, it consists of four functions: estimator_fn, serving_input_fn, train_input_fn, and _generate_input_fn.


A few things to remember when using pre-made estimators:


* For classification problems, make sure that your label (the target variable you want to predict) is processed using categorical encoding, do not use one-hot encoding. Pre-made TensorFlow estimators that perform classification will throw errors. Some custom models using Keras and TensorFlow will train locally but fail when submitted to SageMaker infrastructure; you will get a dimension mismatch error. 


* Make sure you set the correct number of classes for classification problems.


* Make sure you set the proper feature shape in ```serving_input_fn```. Our bag of words model has 500 features because we set the max number of words argument to 500 when building the feature matrix. So pay attention to this line:

```python     feature_spec = {INPUT_TENSOR_NAME: tf.FixedLenFeature(dtype=tf.float32, shape=[500])} ```


* Finally, make sure you have he correct datatypes when generating your training data inputs. You can see in ```python _generate_input_fn``` that the target type and feature type are set:

```python
    filename=os.path.join(training_dir, training_filename), target_dtype=np.int, features_dtype=np.float32)
```

***

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

## Using custom estimators

### TensorFlow addresses the use of custom estimators [here](https://www.tensorflow.org/get_started/custom_estimators).

A ```model_fn``` function implements model training, evaluation, and prediction. SageMaker's [repo](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/tensorflow_abalone_age_predictor_using_keras/tensorflow_abalone_age_predictor_using_keras.ipynb) has a in depth explanation of how these are constructed.



```python
def model_fn(features, labels, mode, params):
   # Logic to do the following:
   # 1. Configure the model via TensorFlow or Keras operations
   # 2. Define the loss function for training/evaluation
   # 3. Define the training operation/optimizer
   # 4. Generate predictions
   # 5. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object
   return EstimatorSpec(mode, predictions, loss, train_op, eval_metric_ops)
   ```

If you are somewhat familiar with machine learning and deep learning, configuring the model, defining loss, and the optimizer may be familiar to you. A few issues to pay attention to:

* Make sure you are passing the correct outputs to your loss function. So predicted class for classification problems. For regression problems you will pass your output through a linear activation and reshape it:

```python

  # Connect the output layer to second hidden layer (no activation fn)
  output_layer = Dense(1, activation='linear')(second_hidden_layer)

  # Reshape output layer to 1-dim Tensor to return predictions
  predictions = tf.reshape(output_layer, [-1])
  
```

* Make sure to create a predictions dictionary with the output you want when serving predictions from your endpoint.


* Make sure to set the feature size in ```python def serving_input_fn(params)```. When we processed our data we tokenized the text, created a bag of words matrix, but we set the maximum number of words argument to 500. Make sure the inputs match this dimension.

```python

    inputs = {INPUT_TENSOR_NAME: tf.placeholder(tf.float32, [None, 500])}
```

* Make sure to set the correct datatypes in the input function ```python def _input_fn(training_dir, training_filename)```. You can see that the label and features have distinct datatypes:

```python filename=os.path.join(training_dir, training_filename), target_dtype=np.int, features_dtype=np.float32)```

* Make sure that your target label you are training on is set as a categorical encoding, which means do not use one-hot encoding. This is true for the pre-made TensorFlow estimators and custom estimators that perform classification. I was able to build and train custom Keras and TensorFlow estimators locally with one-hot encoding, but this would always fail when submitting training to SagerMaker's infrastructure.


***

### Full estimator:

***

```python
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn
from tensorflow.python.estimator.export.export_output import PredictOutput


INPUT_TENSOR_NAME = "inputs"
SIGNATURE_NAME = "serving_default"
LEARNING_RATE = 0.001


def model_fn(features, labels, mode, params):
    
    # 1. Configure the neural net, in this case a very simple two layer network.
    first_hidden_layer = tf.keras.layers.Dense(128, activation='relu', name='first-layer')(features[INPUT_TENSOR_NAME])
    second_hidden_layer = tf.keras.layers.Dense(256, activation='relu')(first_hidden_layer)
    logits = tf.keras.layers.Dense(20)(second_hidden_layer)

    # 1a. This is a classification example we need to find our class predicitons. 
    predicted_classes = tf.argmax(logits, axis=1)

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,},
            export_outputs={SIGNATURE_NAME: PredictOutput({"jobs": predicted_classes})})

    # 2. Define the loss function for training/evaluation using Tensorflow.
    loss = tf.losses.sparse_softmax_cross_entropy(tf.cast(labels, dtype=tf.int32), logits)

    # 3. Define the training operation/optimizer using Tensorflow operation/optimizer.
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=params["learning_rate"],
        optimizer="Adam")

    # 4. Generate predictions as Tensorflow tensors.
    predictions_dict = {"jobs": predicted_classes,
                        "classes": logits}

    # 5. Generate necessary evaluation metrics.
    # Calculate accuracy
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels, predicted_classes)
    }

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)
    

def serving_input_fn(params):
    inputs = {INPUT_TENSOR_NAME: tf.placeholder(tf.float32, [None, 500])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


params = {"learning_rate": LEARNING_RATE}


def train_input_fn(training_dir, params):
    return _input_fn(training_dir, 'post_train.csv')


def eval_input_fn(training_dir, params):
    return _input_fn(training_dir, 'post_test.csv')

def _input_fn(training_dir, training_filename):
    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=os.path.join(training_dir, training_filename), target_dtype=np.int, features_dtype=np.float32)

    return tf.estimator.inputs.numpy_input_fn(
        x={INPUT_TENSOR_NAME: np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)()
```

### Once you have your custom estimator defined you can submit it for training on SageMaker's infrastructure, this will take several minutes:

```python
from sagemaker.tensorflow import TensorFlow

custom_estimator = TensorFlow(entry_point='custom_estimator.py',
                               role=role,
                               training_steps= 1000,                                  
                               evaluation_steps= 100,
                               hyperparameters={'learning_rate': 0.001},
                               train_instance_count=1,
                               train_instance_type='ml.c4.xlarge')

custom_estimator.fit(inputs)
```

### You can then deploy the mode, this will also take several minutes:

```python
custom_predictor = custom_estimator.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
```

### Finally, test the endpoint, i.e. make predictions:

```python
import tensorflow as tf
import numpy as np

prediction_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=os.path.join('data/post_holdout.csv'), target_dtype=np.int, features_dtype=np.float32)

data = prediction_set.data[0]
tensor_proto = tf.make_tensor_proto(values=np.asarray(data), shape=[1, len(data)], dtype=tf.float32)

custom_predictor.predict(tensor_proto)
```

### Make sure to delete the endpoint since we are just testing and do not want to incur charges:

```python
sagemaker.Session().delete_endpoint(custom_predictor.endpoint)
```

