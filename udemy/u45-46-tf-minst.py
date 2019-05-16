
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(type(mnist))
print(mnist.train.images)
print(type(mnist.train.images))
print(mnist.train.num_examples)
print(mnist.test.num_examples)
print(mnist.validation.num_examples)
