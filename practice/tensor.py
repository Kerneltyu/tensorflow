import tensorflow as tf

t1 = tf.constant(1, name='Rank0')

t2 = tf.constant([1,2], name='Rank1')

t3 = tf.constant([[1,2], [3,4]], name='Rank2')

with tf.Session() as sess:
    print(sess.run(t1))
    print(sess.run(t2))
    print(sess.run(t3))
