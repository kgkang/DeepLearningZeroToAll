import tensorflow as tf


# 기본 타입은 tf.float32
# 기본 초기화는 tf.glorot_uniform_initializer
my_variable = tf.get_variable("my_var", [1,2,3])

# dtype과 초기화 방법을 명시할 수 있다.
my_int_var = tf.get_variable("my_int_var", [1,2,3],initializer=tf.zeros_initializer)


# tensor를 초기값으로 줄 수 있다.
# 이경우 shape 인자는 없어야 한다.
other_var = tf.get_variable("other_variable", dtype=tf.int32, initializer=tf.constant([23,42]))


# tf.GraphKeys.TRAINABLE_VARIABLES 에 포함시키지 않으려면..
# 방법 1 : LOCAL_VARIABLES에 포함 시킴
# 방법 2 : trainable=False로 설정
local_1 = tf.get_variable("local_1", shape=(), collections=[tf.GraphKeys.LOCAL_VARIABLES])
local_2 = tf.get_variable("local_2", shape=(), trainable=False)

# 커스텀 collection 생성
tf.add_to_collection("my_collection", local_1)
tf.add_to_collection("my_collection", local_2)

# collection 안의 변수 조회
print(tf.get_collection("my_collection"))
print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

with tf.device("/device:GPU:1"):
    v = tf.get_variable("v", [1])

