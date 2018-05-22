import tensorflow as tf
counter = tf.Variable(0, name='counter')
step_size = tf.constant(1 ,name='step_size')

increment_op = tf.add(counter, step_size)
count_up_op = tf.assign(counter, increment_op)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(count_up_op))
    print(sess.run(count_up_op))
    print(sess.run(count_up_op))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(count_up_op))
    print(sess.run(count_up_op))
    print(sess.run(count_up_op))
