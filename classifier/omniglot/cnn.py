import inputs
import tensorflow as tf

def main():
    train,test = inputs.get_data()
    train_images, train_labels = inputs.train_batch(train)
    train_logits = inference(train_images)
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

#モデルの定義と推論
def inference(inputs, reuse=False):
    with tf.variable_scope('conv1', reuse=reuse):
        weight1 = tf.get_variable(
            'w', [3,3,1,16],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias1 = tf.get_variable(
            'b', shape=[16],
            initializer=tf.zeros_initializer())
        conv1 = tf.nn.conv2d(inputs, weight1, [1,2,2,1], 'VALID')
        out1 = tf.nn.relu(tf.add(conv1, bias1))
    pool1 = tf.nn.max_pool(out1, [1,2,2,1], [1,2,2,1], 'VALID')

    with tf.variable_scope('conv2', reuse=reuse):
        weight2 = tf.get_variable(
            'w', [3,3,16,24],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias2 = tf.get_variable(
            'b', shape=[24],
            initializer=tf.zeros_initializer())
        conv2 = tf.nn.conv2d(pool1, weight2, [1,1,1,1], 'VALID')
        out2 = tf.nn.relu(tf.add(conv2, bias2))
    pool2 = tf.nn.avg_pool(out2, [1,2,2,1], [1,2,2,1], 'VALID')

    with tf.variable_scope('conv3', reuse=reuse):
        weight3 = tf.get_variable(
            'w', [3,3,24,36],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias3 = tf.get_variable(
            'b', shape=[36],
            initializer=tf.zeros_initializer())
        conv3 = tf.nn.conv2d(pool2, weight3, [1,1,1,1], 'VALID')
        out3 = tf.nn.relu(tf.add(conv3, bias3))
    pool3 = tf.nn.max_pool(out3, [1,2,2,1], [1,2,2,1], 'VALID')

    reshape = tf.reshape(pool3, [pool3.get_shape()[0].value, -1])
    with tf.variable_scope('fully_connect', reuse=reuse):
        weight4 = tf.get_variable(
            'w', [5*5*36, 47],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias4 = tf.get_variable(
            'b', shape=[47],
            initializer=tf.zeros_initializer())
        out4 = tf.add(tf.matmul(reshape, weight4), bias4)
    return out4

def loss(labels, logits):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels = labels,
        logits = logits)
    return tf.reduce_mean(cross_entropy)

def training(loss):
    optimizer = tf.train.AdamOptimizer()
    return optimizer.minimize(loss)

if __name__ == '__main__':
    main()
