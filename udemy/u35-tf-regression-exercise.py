
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# csv 데이터 읽어오기
cal_housing = pd.read_csv("../data/cal_housing_clean.csv")
# print(cal_housing.columns)
# ==> Index(['housingMedianAge', 'totalRooms', 'totalBedrooms', 'population',
#        'households', 'medianIncome', 'medianHouseValue'],
#       dtype='object')
# ** 켈리포니아 주택 조사 : 블록별 주택 통계, 한블록에 있는 주택수, 등
# ** housingMedianAge : 블록내 주택들의 연령(중위값)
# ** totalRooms : 블록내 주택들의 총 방수
# ** totalBedrooms : 블록내 주택들의 총 침실수
# ** population : 블록내 주택들이 총 인구
# ** households : 블록내 총 주택수? (가구수)
# ** medianIncome : 블록 내 가구의 수입 (중위값)
# ** medianHouseValue : 블록 내 주택 가격 (중위값)
print(cal_housing.head())
# print(type(cal_housing))



# 사용할 전체 데이터를 만들고,
# 여기에서 X,Y 데이터로 분리해서 만들기.
x_data = cal_housing.drop('medianHouseValue', axis=1)
y_data = cal_housing['medianHouseValue']
print(x_data.head())
# print(y_data.head())

# minmax scaler
x_data = x_data.apply(lambda x: (x-x.min())/(x.max()-x.min()))
print(x_data.head())


# x,y 데이터를 train, test 데이터 분리하기
x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_data, test_size=0.3, random_state=101)
print(x_train.head())

##
# scaler = MinMaxScaler(copy=True, feature_range=(0,1))
# scaler = scaler.fit(x_train)
# x_train = pd.DataFrame(data=scaler.transform(x_train), columns=x_train.columns, index=x_train.index)
# x_eval  = pd.DataFrame(data=scaler.transform(x_eval), columns=x_eval.columns, index=x_eval.index)
# print(x_train.head())
# print(x_eval.head())


# feature column을 선언한다.
# ['housingMedianAge', 'totalRooms', 'totalBedrooms', 'population', 'households', 'medianIncome', 'medianHouseValue']
num_age = tf.feature_column.numeric_column('housingMedianAge')
num_rooms = tf.feature_column.numeric_column('totalRooms')
num_bedrooms = tf.feature_column.numeric_column('totalBedrooms')
num_population = tf.feature_column.numeric_column('population')
num_households = tf.feature_column.numeric_column('households')
num_income = tf.feature_column.numeric_column('medianIncome')
# num_housevalue = tf.feature_column.numeric_column('medianHouseValue')

feature_cols = [num_age, num_rooms, num_bedrooms, num_population, num_households, num_income]
# print (feature_cols)
# Model을 정의한다.
model = tf.estimator.DNNRegressor(feature_columns=feature_cols, hidden_units=[6,6,6])

# Train Input function을 정의한다.
train_input_func = tf.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)

# Model을 훈련시킨다.
model.train(input_fn=train_input_func, steps=10000)

# Test Input function을 정의한다.
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=x_eval, y=y_eval, batch_size=10, num_epochs=1000, shuffle=False)
eval_metric = model.evaluate(input_fn=eval_input_func, steps=1000)
print(eval_metric)

# 예측
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=x_eval, batch_size=4, num_epochs=1, shuffle=False)
pred_model = model.predict(pred_input_func)
pred_list = list(pred_model)

pred_final = []
for pred in pred_list:
    pred_final.append(pred['predictions'])

print(type(pred_model), type(pred_list), type(pred_final), type(y_eval))


from sklearn.metrics import mean_squared_error
# mean_squared_error에 다른 타입의 인자를 넘겨도 되나?
# y_eval은 pandas.core.series.Series, pred_final은 list임...
rmse = mean_squared_error(y_eval, pred_final)**0.5
print(rmse)








