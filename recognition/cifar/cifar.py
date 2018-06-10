from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inputs
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    #preprocessが必要
    input_layer = tf.reshape(features["x"], [-1, 32, 32, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer, filters=32, kernel_size=[5,5], padding="same", activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
    conv2 = tf.layers.conv2d(
        inputs=pool1, filters=64, kernel_size=[2,2], padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 8*8*64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.5, training= mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dropout, units=10, activation=tf.nn.softmax)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    #lossの計算，関数で行ってもいけそう
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels = labels, predictions=predictions["classes"]
            )
        }
        return tf.estimator.EstimatorSpec(
            mode = mode, loss=loss, eval_metric_ops=eval_metric_ops
        )

def main(unused_argv):
    train, test = inputs.get_data()
    train_images, train_labels = inputs.all_batch(train)
    test_images, test_labels = inputs.all_batch(test)
    cifar_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/cifar_convert_model"
    )
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x" : train_images},
        y = train_labels,
        batch_size=32,
        num_epochs=None,
        shuffle=True
    )
    cifar_classifier.train(input_fn=train_input_fn, steps=20000, hooks=[logging_hook])
    #print(train_images)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_images},
        y=test_labels,
        num_epochs=1,
        shuffle=False
    )
    eval_results = cifar_classifier.evaluate(input_fn = eval_input_fn)
    print(eval_results)

if __name__ == '__main__':
    tf.app.run()
