import tensorflow as tf

# Feature 데이터
features = {
    'sales' : [[5],[10],[8],[9]],
    'department' : ['sport','sport','gardening','gardening']
}

column_sales = tf.feature_column.numeric_column('sales')
column_department = tf.feature_column.categorical_column_with_vocabulary_list('department', ['sport','gardening'])
column_department = tf.feature_column.indicator_column(column_department)
# column_department = tf.feature_column.embedding_column(column_department)

feature_columns = [column_sales, column_department]

inputs = tf.feature_column.input_layer(features, feature_columns)

# 그래프 초기화
# Feature column은 내부에 tf.contrib.lookup을 사용해서, tf.tables_initializer로 초기화함.

var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()

sess = tf.Session()

sess.run((var_init, table_init))
print(sess.run(inputs))
