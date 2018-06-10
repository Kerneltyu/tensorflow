import inputs
import tensorflow as tf
import numpy as np

def main(argv=None):
    train, test = inputs.get_data() #inputsファイル用意
    train_images, train_labels = inputs.train_batch(train)
    train_logits = inference(train_images) #推論
    losses = loss(train_labels, train_logits)
    train_op = training(losses)
    test_images, test_labels = inputs.test_batch(test)
    test_logits = inference(test_images,reuse=True)
    correct_prediction = tf.equal(tf.argmax(test_logits, 1), tf.to_int64(test_labels))
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction))

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        sess.run(tf.global_variables_initializer())

        for i in range(300):
            _, loss_value, accuracy_value = sess.run([train_op, losses, accuracy])
            print('step {:3d}: {:5f}（{:3f}）'.format(i+1, loss_value, accuracy_value * 100.0))

        coord.request_stop()
        coord.join(threads)

def inference(inputs, reuse=False):
    with tf.variable_scope('conv1'):
        #filterの指定
        weight1 = tf.get_variable(
            'w', [11,11,3,96],
            initializer = tf.truncated_normal_initializer(stddev=0.1)
        )
        bias1 = tf.get_variable(
            'b', shape=[96],
            initializer = tf.zeros_initializer()
        )
        conv1 = tf.nn.conv2d(inputs, weight1, strides=[1,4,4,1], padding='SAME')
        out1 = tf.nn.relu(tf.add(conv1, bias1))
    norm1 = tf.nn.local_response_normalization(out1)
    pool1 = tf.nn.max_pool(norm1, ksize=[3,3,96], strides=[1,2,2,1], padding='SAME')

    with tf.variable_scope('conv2'):
        weight2 = tf.get_variable(
            'w', [5,5,96,256],
            initializer = tf.truncated_normal_initializer(stddev=0.1)
        )
        bias2 = tf.get_variable(
            'b', shape=[256],
            initializer=tf.zeros_initializer()
        )
        conv2 = tf.nn.conv2d(pool1, weight2, [1,1,1,1], [1,1,1,1], 'VALID')
        out2 = tf.nn.relu(tf.add(conv2, bias2))
    norm2 = tf.nn.local_response_normalization(out2)
    pool2 = tf.nn.max_pool(norm2, ksize=[3,3,256], strides=[1,2,2,1])
    with tf.variable_scope('conv3'):
        weight3 = tf.get_variable(
            'w', [3,3,256,384],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        bias3 = tf.get_variable(
            'b', [384],
            initializer = tf.zeros_initializer()
        )
        conv3 = tf.nn.conv2d(pool2, weight3, [1,1,1,1], [1,1,1,1], 'VALID')
        out3 = tf.nn.relu(tf.add(conv3, bias3))
    with tf.variable_scope('conv4'):
        weight4 = tf.get_variable(
            'w', [3,3,384,384],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        bias4 = tf.get_variable(
            'b', [384],
            initializer = tf.zeros_initializer()
        )
        conv4 = tf.nn.conv2d(out3, weight4, [1,1,1,1], [1,1,1,1], 'VALID')
        out4 = tf.nn.relu(tf.add(conv4, bias4))
    with tf.variable_scope('conv5'):
        weight5 = tf.get_variable(
            'w', [3,3,384,256],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        bias5 = tf.get_variable(
            'b', [256],
            initializer = tf.zeros_initializer()
        )
        conv4 = tf.nn.conv2d(out4, weight5, [1,1,1,1], [1,1,1,1], 'VALID')
        out4 = tf.nn.relu(tf.add(conv5, bias5))
    pool3 = tf.nn.max_pool(out4,padding=[1,1,1,1],strides=[1,2,2,1])
    reshape = tf.reshape(pool3, [pool3.get_shape()[0].value, -1])
    with tf.variable_scope('fully_connect', reuse=reuse):
        weight6 = tf.get_variable(
            'w', [6*6*256,4096],
            initializer=tf.zeros_initializer()
        )
        bias6 = tf.get_variable(
            'b', shape=[4096],
            initializer=tf.zeros_initializer()
        )
        out6 = tf.add(tf.matmul(reshape, weight6), bias6)
        out6_drop = tf.nn.dropout(out6, 0.5)
    with tf.variable_scope('fully_connect', reuse=reuse):
        weight7 = tf.get_variable(
            'w', [4096],
            initializer=tf.zeros_initializer()
        )
        bias7 = tf.get_variable(
            'b', shepe=[4096],
            initializer=tf.zeros_initializer()
        )
        out7 = tf.add(tf.matmul(out6_drop, weight7), bias7)
        out7_drop = tf.nn.dropout(out7, 0.5)
    with tf.variable_scope('fully_connect', reuse=reuse):
        weight8 = tf.get_variable(
            'w', [4096],
            initializer=tf.zeros_initializer()
        )
        bias8 = tf.get_variable(
            'b', shape=[4096],
            initializer=tf.zeros_initializer()　zazsacfdsabvcx 　
        )
        out8 = tf.nn.softmax(tf.matmul(out7_drop, weight8) + bias8)
    return out8

def loss(labels, logits):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits = logits
    )
    return tf.reduce_mean(cross_entropy)

def training(loss):
    optimizer = tf.train.AdamOptimizer()
    return optimizer.minimize(loss)

if __name__ == '__main__':
    tf.app.run()
