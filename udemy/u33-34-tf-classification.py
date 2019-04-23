
import pandas as pd
import tensorflow as tf
import matplotlib.pylab as plt

# # parent directory를 search path에 추가한다.
# import sys,os
# # print(os.path.join(os.pardir, "data"))
# sys.path.append(os.path.join(os.pardir, "data"))
# print(sys.path)

# pima 인디언 부족의 당료질환 데이터, 총 768개의 데이터로 구성됨.
# 'Class'가 당료인지 아닌지를 의미한다.
# Cols => ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
# #        'Insulin', 'BMI', 'Pedigree', 'Age', 'Class', 'Group']
diabetes = pd.read_csv('../data/pima-indians-diabetes.csv')
# pd dataframe의 첫 5라인을 출력
# print(diabetes.head())
# Columns Index 이름을 출력
# print(diabetes.columns)

# 숫자 컬럼을 정규화 시킨다.
cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
       'Insulin', 'BMI', 'Pedigree']
diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x-x.min())/(x.max()-x.min()))
print(diabetes.head())

# feature colume과 numeric column을 지정한다.
num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

# numeric column을 categorical 로 전환한다.
# diabetes['Age'].hist(bins=20)
# plt.show()
age_bucket = tf.feature_column.bucketized_column(age, boundaries=[20,30,40,50,60,70,80])

assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group',['A','B','C','D'])
# 분류 항목이 많을 경우, 자동으로 생성하게 해준다.
# assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group',hash_bucket_size=10)

feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, diabetes_pedigree, assigned_group, age_bucket]


# 훈련 데이터 분리
# 'Class' 레이블 데이터를 삭제한다.
x_data = diabetes.drop('Class', axis=1)
x_label = diabetes['Class']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, x_label, test_size=0.3, random_state=101)

input_func = tf.estimator.inputs.pandas_input_fn(x=x_train, y=y_train,
                                                 batch_size=10, num_epochs=1000,
                                                 shuffle=True)

# 평가
model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)
model.train(input_fn=input_func, steps=1000)

eval_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test, y=y_test,
                                                 batch_size=10, num_epochs=1,
                                                 shuffle=False)

results = model.evaluate(eval_input_func)
print(results)

# 예측 수행
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test,
                                                      batch_size=10, num_epochs=1,
                                                      shuffle=False)
predictions = model.predict(pred_input_func)
my_pred = list(predictions)
print(my_pred)



# DNN Model (Dense Neural Network)

# dnn 모델에서는 categorical_column을 embedding_column로 변환해서 입력해줘야 한다.
embedded_group_col = tf.feature_column.embedding_column(assigned_group, dimension=4)
feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, diabetes_pedigree, embedded_group_col, age_bucket]
# input_func = tf.estimator.inputs.pandas_input_fn(x_train, y_train, batch_size=10, num_epochs=1000, shuffle=True)


# 10,10,10개의 nuron으로 구성된 3개층의 dnn
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10], feature_columns=feat_cols,n_classes=2)
dnn_model.train(input_fn=input_func, steps=1000)

# eval_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
dnn_result = dnn_model.evaluate(eval_input_func)
print(dnn_result)
