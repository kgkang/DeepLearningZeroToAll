import tensorflow as tf


with tf.Graph().as_default(), tf.Session() as sess:
    dice1 = tf.Variable(tf.random.uniform([10,1],minval=1,maxval=7,dtype=tf.int32))
    dice2 = tf.Variable(tf.random.uniform([10,1],minval=1,maxval=7,dtype=tf.int32))

    sum = tf.add(dice1,dice2)

    # We've got three separate 10x1 matrices. To produce a single
    # 10x3 matrix, we'll concatenate them along dimension 1.
    out = tf.concat(values=[dice1, dice2, sum], axis=1)
    # out = tf.concat(values=[dice1, dice2, sum], axis=0)

    sess.run(tf.global_variables_initializer())
    # print(dice1.eval())
    # print(dice2.eval())
    # print(sum.eval())
    print(out.eval())