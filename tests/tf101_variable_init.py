import tensorflow as tf

g = tf.Graph()
with g.as_default(), tf.Session() as sess:
    # Create a variable with the initial value 3.
    v = tf.Variable([3])

    # Create a variable of shape [1], with a random initial value,
    # sampled from a normal distribution with mean 1 and standard deviation 0.35.
    w = tf.Variable(tf.random_normal([1], mean=1.0, stddev=0.35))

    initializer = tf.global_variables_initializer()
    sess.run(initializer)

    try:
        print(v.eval())
    except tf.errors.FailedPreconditionError as e:
        print("Caught expected error: ", e)

    assignment = tf.assign(v,[7])
    # print(assignment.eval())
    print(sess.run(assignment))
