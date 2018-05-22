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
    '''
    Args:
        inputs: [batch_size, height, width, channels]のTensor
        reuse: 変数を再利用するか否か
    Returns:
        推論結果の[batch_size, 47]のTensor
    '''
    reshaped = tf.reshape(inputs, [inputs.get_shape()[0].value, -1])
    with tf.variable_scope('fully_connect1', reuse=reuse):
        weight1 = tf.get_variable(
            'w', [105 * 105, 100],
            initializer=tf.truncated_normal_initializer(stddev=0.01))
        bias1 = tf.get_variable(
            'b', shape=[100],
            initializer=tf.zeros_initializer())
        out1 = tf.nn.relu(tf.add(tf.matmul(reshaped, weight1), bias1))

    with tf.variable_scope('fully_connect2', reuse=reuse):
        weight2 = tf.get_variable(
            'w', [100, 47],
            initializer = tf.truncated_normal_initializer(stddev=0.1))
        bias2 = tf.get_variable(
            'b', [47],
            initializer = tf.zeros_initializer())
        out2 = tf.add(tf.matmul(out1, weight2), bias2)
    return out2

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
