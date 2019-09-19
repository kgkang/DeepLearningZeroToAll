import tensorflow as tf


# 참고링크 :  https://www.tensorflow.org/guide/low_level_intro#layers

x = tf.placeholder(tf.float32, shape=[None, 3]) # 입력 정의
# model = tf.layers.Dense(units=1) # Layer가 1층, 1개인 Dense 모델
# y = model(x)  # Layer에 input을 연결

y = tf.layers.dense(x, units=1)  # shortcut 함수를 이용해서, 위 2줄을 1출로 바꿀 수 있음.

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
print(sess.run(y, {x:[[1,2,3],[4,5,6]]})) # 입력에 'feed_dict='을 생략할 수 있음.
print(sess.run(y, feed_dict={x:[[7,8,9],[10,11,12]]}))

sess.close()