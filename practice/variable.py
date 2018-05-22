import tensorflow as tf

counter = tf.Variable(0, name="counter")
step_size = tf.constant(1, name="step_size")

#演算の定義
increment_op = tf.add(counter, step_size)
#代入の定義
count_up_op = tf.assign(counter, increment_op)

with tf.Session() as sess:
    #変数を用いる場合ははじめに初期化が必要
    sess.run(tf.global_variables_initializer())
    print(sess.run(counter))
    print(sess.run(increment_op))
    print(sess.run(count_up_op))
    print(sess.run(count_up_op))
    print(sess.run(count_up_op))
    print(sess.run(counter))
