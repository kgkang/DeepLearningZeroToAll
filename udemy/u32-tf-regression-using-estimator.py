
# https://www.udemy.com/complete-guide-to-tensorflow-for-deep-learning-with-python/learn/lecture/8695996#overview

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pylab as plt

# 전체 데이터 생성
x_data = np.linspace(0,10.0, 1000000)
noise = np.random.randn(len(x_data))
y_data = (5.1 * x_data) + 2.5 + noise

x_df = pd.DataFrame(data=x_data, columns=['X data'])
y_df = pd.DataFrame(data=y_data, columns=['Y'])
my_data = pd.concat([x_df,y_df],axis=1)


# 훈련 및 평가 데이터 생성
# 데이터가 미리 준비되어 있으면 필요 없음.
from sklearn.model_selection import train_test_split
x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_data, test_size=0.3, random_state=101)
print(type(x_train), x_train.shape)
print(type(x_eval), x_eval.shape)


# 모델과 입력형상를 정의를 편하게 하기 위해
# 모델은 estimator 모델 중에서 선정하고
# 입력형상은 (placeholder격의) feature_column을 정의함.
feat_cols = [tf.feature_column.numeric_column('x',shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

# 입력 함수를 통해서 모델에 데이터를 주입
# 입력 함수 정의하는 부분에서 batch_size를 지정하게 되어 있음.
input_func = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size=8, num_epochs=None, shuffle=True)
train_input_func = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size=8, num_epochs=1000, shuffle=False)
eval_input_func = tf.estimator.inputs.numpy_input_fn({'x': x_eval}, y_eval, batch_size=8, num_epochs=1000, shuffle=False)

estimator.train(input_fn=input_func, steps=1000)
train_metric = estimator.evaluate(input_fn=train_input_func, steps=1000)
eval_metric = estimator.evaluate(input_fn=eval_input_func, steps=1000)
print(train_metric)
print(eval_metric)

# 결과 regression 라인을 그리기 위한 데이터
brand_new_data = np.linspace(0,10,10)
input_func_predict = tf.estimator.inputs.numpy_input_fn({'x':brand_new_data}, shuffle=False)
predictions = []
for pred in estimator.predict(input_fn=input_func_predict):
    predictions.append(pred['predictions'])

print(type(predictions))

my_data.sample(n=250).plot(kind='scatter', x="X data", y="Y")
plt.plot(brand_new_data,predictions,'r')
plt.show()



