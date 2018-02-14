
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn
from tensorflow.python.estimator.export.export_output import PredictOutput


INPUT_TENSOR_NAME = "inputs"
SIGNATURE_NAME = "serving_default"
LEARNING_RATE = 0.001


def model_fn(features, labels, mode, params):
    
    first_hidden_layer = tf.keras.layers.Dense(128, activation='relu', name='first-layer')(features[INPUT_TENSOR_NAME])
    second_hidden_layer = tf.keras.layers.Dense(256, activation='relu')(first_hidden_layer)
    logits = tf.keras.layers.Dense(20)(second_hidden_layer)

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
    # Calculate root mean squared error as additional eval metric
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
