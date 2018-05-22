import tensorflow as tf

x = tf.placeholder(tf.int32, name='x')
y = tf.placeholder(tf.int32, name='y')

add_op = tf.add(x,y)
mul_op = tf.multiply(x,y)

with tf.Session() as sess:
    print(sess.run(add_op, feed_dict={x:1, y:2}))
    print(sess.run(mul_op, feed_dict={x:1, y:2}))

    print(sess.run(add_op, feed_dict={x:100, y:200}))
    print(sess.run(mul_op, feed_dict={x:100, y:200}))
