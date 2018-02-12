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


