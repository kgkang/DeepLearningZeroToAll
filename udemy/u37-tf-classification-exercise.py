
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 켈리포니아 인구 조사 ?
# ==> 총 32561개의 row, 14개의 컬럼
# 1 'age' : 연령,
# 2 'workclass'
# 3 'education'
# 4 'education_num'
# 5 'marital_status'
# 6 'occupation'
# 7 'relationship'
# 8 'race'
# 9 'gender'
# 10 'capital_gain'
# 11 'capital_loss'
# 12 'hours_per_week'
# 13 'native_country'
# 14 'income_bracket'

census = pd.read_csv("../data/census_data.csv")
# print(type(census))
# ==> <class 'pandas.core.frame.DataFrame'>
# 데이터프래임의 정보를 확인하는 API ?
# 데이터프래임의 컬럼 데이터에 대한 label을 확인하는 API ?
print(census.columns)
# 데이터프래임 컬럼의 데이터 종류를 확인할 수 있는 API ?
print(census['income_bracket'].unique())
# print(census.head())
# print(census)

print(census.describe().transpose())
#                   count         mean          std  ...   50%   75%      max
# age             32561.0    38.581647    13.640433  ...  37.0  48.0     90.0
# education_num   32561.0    10.080679     2.572720  ...  10.0  12.0     16.0
# capital_gain    32561.0  1077.648844  7385.292085  ...   0.0   0.0  99999.0
# capital_loss    32561.0    87.303830   402.960219  ...   0.0   0.0   4356.0
# hours_per_week  32561.0    40.437456    12.347429  ...  40.0  45.0     99.0

# income_bracket 컬럼은 string 타입이다. ' <=50K',' >50K'를 0,1로 변환한다.
census['income_bracket'] = census['income_bracket'].apply(lambda x: 0 if x == ' <=50K' else 1)
# census['income_bracket'] = census['income_bracket'].apply(lambda x: int(x == ' >50K') )
# print(census.head())


# Split train, test data
x_data = census.drop('income_bracket', axis=1)
y_data = census['income_bracket']
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=101)
# print(x_train.head())
# print(y_train.head())

# ## 모델 설계 : feature 정의 (데이터 이름 참조), 모델 선택
# Feature columns
ft_age = tf.feature_column.numeric_column('age')
ft_workclass =tf.feature_column.categorical_column_with_hash_bucket('workclass', hash_bucket_size=10)
ft_education = tf.feature_column.categorical_column_with_hash_bucket('education', hash_bucket_size=10)
ft_education_num = tf.feature_column.numeric_column('education_num')
ft_marital_status = tf.feature_column.categorical_column_with_hash_bucket('marital_status', hash_bucket_size=10)
ft_occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation', hash_bucket_size=30)
ft_relationship = tf.feature_column.categorical_column_with_hash_bucket('relationship', hash_bucket_size=10)
ft_race = tf.feature_column.categorical_column_with_hash_bucket('race', hash_bucket_size=10)
ft_gender = tf.feature_column.categorical_column_with_hash_bucket('gender', hash_bucket_size=10)
ft_capital_gain = tf.feature_column.numeric_column('capital_gain')
ft_capital_loss = tf.feature_column.numeric_column('capital_loss')
ft_hours_per_week = tf.feature_column.numeric_column('hours_per_week')
ft_native_country = tf.feature_column.categorical_column_with_hash_bucket('native_country', hash_bucket_size=100)


# 모델과 Feature를 설정
#
ft_cols = [ft_age, ft_workclass, ft_education, ft_education_num, ft_marital_status, ft_occupation,
           ft_relationship, ft_race, ft_gender, ft_capital_gain, ft_capital_loss, ft_hours_per_week,
           ft_native_country]

model = tf.estimator.LinearClassifier(feature_columns=ft_cols, n_classes=2)
# model = tf.estimator.DNNClassifier(hidden_unit=[10,10,10], feature_columns=ft_cols, n_classes=2)

# Input Function 정의
train_input_func = tf.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size = 10, num_epochs = 3000, shuffle=True)
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
test_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test, batch_size=10, num_epochs=1, shuffle=False)

# 모델 학습
model.train(input_fn=train_input_func, steps=3000)

# 모델 평가
model_evaluate = model.evaluate(input_fn=eval_input_func)
print(model_evaluate)

# 모델 예측
model_predict = model.predict(input_fn=test_input_func)
predictions = list(model_predict)
final_pred = [pred['class_ids'][0] for pred in predictions]
print(final_pred)

# 통계 출력
from sklearn.metrics import classification_report
print(classification_report(y_test, final_pred))
