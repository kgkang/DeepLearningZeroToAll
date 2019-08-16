import tensorflow as tf

# Create some variables
v1 = tf.get_variable("v1", shape=[3], initializer=tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer=tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    inc_v1.op.run()
    dec_v2.op.run()
    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in path : %s " % save_path)



tf.reset_default_graph()

v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "./tmp/model.ckpt")
    print("Model restored.")
    print("v1 : %s" % v1.eval())
    print("v2 : %s" % v2.eval())


tf.reset_default_graph()

# Create some variables.

v1 = tf.get_variable("v1", [3], initializer= tf.zeros_initializer)
v2 = tf.get_variable("v2", [5], initializer= tf.zeros_initializer)

# Add ops to save and restore only 'v2' using the name "v2"
saver = tf.train.Saver({"v2": v2})

with tf.Session() as sess:
    v1.initializer.run()
    saver.restore(sess, "./tmp/model.ckpt")

    print("v1 : %s" % v1.eval())
    print("v2 : %s" % v2.eval())