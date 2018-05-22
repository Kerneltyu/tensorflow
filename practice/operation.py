import tensorflow as tf
t1 = tf.constant(21, name='Tensor1')
t2 = tf.constant([1,2], name='Tensor2')

add_op = tf.add(t1, t2)
mul_op = tf.multiply(t1, t2)

with tf.Session() as sess:
    print("{} + {} = {}".format(sess.run(t1), sess.run(t2), sess.run(add_op)))
    print("{} * {} = {}".format(sess.run(t1), sess.run(t2), sess.run(mul_op)))
