import tensorflow as tf

# 데이터 정의
x = tf.constant([[1],[2],[3],[4]], dtype=tf.float32)
y_true = tf.constant([[0],[-1],[-2],[-3]], dtype=tf.float32)

# 모델 정의
linear_model = tf.layers.Dense(units=1)

# 손실 정의
y_pred = linear_model(x)
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

# 훈련 정의 : 손실을 최소화 시키는 방법 점의
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train = optimizer.minimize(loss)

# 세션 생성
init = tf.global_variables_initializer()
sess = tf.Session()

# 훈련 실행
sess.run(init)

for i in range(100):
    _, loss_value = sess.run((train, loss))
    print(loss_value)


print(sess.run(y_pred))
# print(y_pred.eval())