import tensorflow as tf

tf.reset_default_graph()

# tf101_save_restore.py를 먼저 실행한다.
# Create some variables.

v3 = tf.get_variable("v3", [3], initializer= tf.zeros_initializer)
v2 = tf.get_variable("v2", [5], initializer= tf.zeros_initializer)

init_op = tf.global_variables_initializer()

# Add ops to save and restore only 'v3' using the name "v1"
# .save() 실행시, v3 변수를 "v1" 이라는 키로 저장하거나,
# .restore() 실행시, "v1" 키로 저장된 데이터를 v3 변수로 복원한다.
saver = tf.train.Saver({"v1": v3})

with tf.Session() as sess:
#    v1.initializer.run()
    sess.run(init_op)
    print("v1 : %s" % v3.eval())
    print("v2 : %s" % v2.eval())
    saver.restore(sess, "./tmp/model.ckpt")

    print("v1 : %s" % v3.eval())
    print("v2 : %s" % v2.eval())