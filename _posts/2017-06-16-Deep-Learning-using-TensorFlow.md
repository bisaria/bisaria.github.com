---
layout: post
title: "Deep Learning using TensorFlow"
date: "June 16, 2017"
category : deeplearning
tagline: ""
tags : [udacity, deeplearning, mooc]
---
{% include JB/setup %}

## Udacity: Deep Learning Assignment 3

Previously in 2_fullyconnected.ipynb, you trained a logistic regression and a neural network model.

The goal of this assignment is to explore regularization techniques.



```python
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
```

First reload the data we generated in 1_notmnist.ipynb.


```python
pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)
```

    Training set (200000, 28, 28) (200000,)
    Validation set (10000, 28, 28) (10000,)
    Test set (10000, 28, 28) (10000,)
    

Reformat into a shape that's more adapted to the models we're going to train:

* data as a flat matrix,
* labels as float 1-hot encodings.


```python
image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
```

    Training set (200000, 784) (200000, 10)
    Validation set (10000, 784) (10000, 10)
    Test set (10000, 784) (10000, 10)
    


```python
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
```

## Problem 1

Introduce and tune L2 regularization for both logistic and neural network models. Remember that L2 amounts to adding a penalty on the norm of the weights to the loss. In TensorFlow, you can compute the L2 loss for a tensor t using nn.l2_loss(t). The right amount of regularization should improve your validation / test accuracy.

#### Logistic Regression


```python
train_subset = 10000
beta = 0.01

graph = tf.Graph()
with graph.as_default():

  # Input data.
  # Load the training, validation and test data into constants that are
  # attached to the graph.
  tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
  tf_train_labels = tf.constant(train_labels[:train_subset])
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  # These are the parameters that we are going to be training. The weight
  # matrix will be initialized using random values following a (truncated)
  # normal distribution. The biases get initialized to zero.
  weights = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.
  # We multiply the inputs with the weight matrix, and add biases. We compute
  # the softmax and cross-entropy (it's one operation in TensorFlow, because
  # it's very common, and it can be optimized). We take the average of this
  # cross-entropy across all training examples: that's our loss.
  logits = tf.matmul(tf_train_dataset, weights) + biases
  
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  # L2 Regularization
  regularizer = tf.nn.l2_loss(weights)
  # loss after using L2 Regularization
  loss = tf.reduce_mean(loss + beta * regularizer)

  # Optimizer.
  # We are going to find the minimum of this loss using gradient descent.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  # These are not part of training, but merely here so that we can report
  # accuracy figures as we train.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
```


```python
num_steps = 801

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases. 
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    # Run the computations. We tell .run() that we want to run the optimizer,
    # and get the loss value and the training predictions returned as numpy
    # arrays.
    _, l, predictions = session.run([optimizer, loss, train_prediction])
    if (step % 100 == 0):
      print('Loss at step %d: %f' % (step, l))
      print('Training accuracy: %.1f%%' % accuracy(
        predictions, train_labels[:train_subset, :]))
      # Calling .eval() on valid_prediction is basically like calling run(), but
      # just to get that one numpy array. Note that it recomputes all its graph
      # dependencies.
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
```

    Initialized
    Loss at step 0: 49.114323
    Training accuracy: 7.2%
    Validation accuracy: 11.5%
    Loss at step 100: 12.039999
    Training accuracy: 74.5%
    Validation accuracy: 72.5%
    Loss at step 200: 4.566074
    Training accuracy: 79.3%
    Validation accuracy: 76.9%
    Loss at step 300: 2.010421
    Training accuracy: 82.5%
    Validation accuracy: 79.6%
    Loss at step 400: 1.139898
    Training accuracy: 83.9%
    Validation accuracy: 81.1%
    Loss at step 500: 0.840110
    Training accuracy: 84.5%
    Validation accuracy: 81.7%
    Loss at step 600: 0.735663
    Training accuracy: 84.7%
    Validation accuracy: 81.9%
    Loss at step 700: 0.698938
    Training accuracy: 84.8%
    Validation accuracy: 82.0%
    Loss at step 800: 0.685928
    Training accuracy: 84.9%
    Validation accuracy: 82.1%
    Test accuracy: 88.8%
    

For beta = 0.01 : Accuracy Score on Test Set:  88.8

#### Neural Network with L2 Regularization
    * Single Layer
    
    Here batch_size for training is considerd as 128, which means the training samples (200000 in this exercise) will been randomly divided into 200000/128 batches with 128 samples in each batch. In other words, every complete training round will have 200000/128 steps.
    
    An epoch is considered as the number of steps when all of the training data has been passed through the learning model once, which will be 200000/128 steps.


```python
num_nodes = 1024
batch_size = 128
beta = 0.01

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  weights_1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_nodes]))
  biases_1 = tf.Variable(tf.zeros([num_nodes]))
  weights_2 = tf.Variable(
    tf.truncated_normal([num_nodes, num_labels]))
  biases_2 = tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.
  relu_layer = tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)
  logits = tf.matmul(relu_layer, weights_2) + biases_2
  
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  # L2 Regularization
  regularizer = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2)
  # loss after using L2 Regularization
  loss = tf.reduce_mean(loss + beta * regularizer)
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights_1) + biases_1), weights_2) + biases_2)
  test_prediction = tf.nn.softmax(
      tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1), weights_2) + biases_2)
```


```python
num_steps = 10001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
```

    Initialized
    Minibatch loss at step 0: 3513.550781
    Minibatch accuracy: 10.2%
    Validation accuracy: 25.3%
    Minibatch loss at step 500: 21.248220
    Minibatch accuracy: 87.5%
    Validation accuracy: 83.9%
    Minibatch loss at step 1000: 0.948650
    Minibatch accuracy: 81.2%
    Validation accuracy: 83.2%
    Minibatch loss at step 1500: 0.586573
    Minibatch accuracy: 87.5%
    Validation accuracy: 83.6%
    Minibatch loss at step 2000: 0.611937
    Minibatch accuracy: 90.6%
    Validation accuracy: 83.4%
    Minibatch loss at step 2500: 0.710213
    Minibatch accuracy: 84.4%
    Validation accuracy: 83.2%
    Minibatch loss at step 3000: 0.765545
    Minibatch accuracy: 82.0%
    Validation accuracy: 83.3%
    Minibatch loss at step 3500: 0.767605
    Minibatch accuracy: 82.0%
    Validation accuracy: 83.7%
    Minibatch loss at step 4000: 0.662750
    Minibatch accuracy: 86.7%
    Validation accuracy: 83.9%
    Minibatch loss at step 4500: 0.688409
    Minibatch accuracy: 86.7%
    Validation accuracy: 83.4%
    Minibatch loss at step 5000: 0.697968
    Minibatch accuracy: 84.4%
    Validation accuracy: 83.7%
    Minibatch loss at step 5500: 0.786071
    Minibatch accuracy: 82.0%
    Validation accuracy: 82.6%
    Minibatch loss at step 6000: 0.824578
    Minibatch accuracy: 78.9%
    Validation accuracy: 82.8%
    Minibatch loss at step 6500: 0.617071
    Minibatch accuracy: 85.9%
    Validation accuracy: 83.8%
    Minibatch loss at step 7000: 0.803146
    Minibatch accuracy: 78.9%
    Validation accuracy: 83.5%
    Minibatch loss at step 7500: 0.874642
    Minibatch accuracy: 79.7%
    Validation accuracy: 83.8%
    Minibatch loss at step 8000: 0.930554
    Minibatch accuracy: 76.6%
    Validation accuracy: 83.0%
    Minibatch loss at step 8500: 0.663493
    Minibatch accuracy: 85.2%
    Validation accuracy: 83.1%
    Minibatch loss at step 9000: 0.836120
    Minibatch accuracy: 81.2%
    Validation accuracy: 83.1%
    Minibatch loss at step 9500: 0.740908
    Minibatch accuracy: 85.9%
    Validation accuracy: 83.4%
    Minibatch loss at step 10000: 0.804774
    Minibatch accuracy: 79.7%
    Validation accuracy: 83.5%
    Test accuracy: 90.6%
    

## Problem 2

Let's demonstrate an extreme case of overfitting. Restrict your training data to just a few batches. What happens?


##### Considering only 500 training data, thereby, restricting number of batches


```python
num_steps = 10001

train_dataset_2 = train_dataset[:500,:]
train_labels_2 = train_labels[:500]

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels_2.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset_2[offset:(offset + batch_size), :]
    batch_labels = train_labels_2[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
```

    Initialized
    Minibatch loss at step 0: 3448.132324
    Minibatch accuracy: 10.9%
    Validation accuracy: 28.4%
    Minibatch loss at step 500: 21.069828
    Minibatch accuracy: 100.0%
    Validation accuracy: 78.3%
    Minibatch loss at step 1000: 0.490235
    Minibatch accuracy: 99.2%
    Validation accuracy: 79.4%
    Minibatch loss at step 1500: 0.300318
    Minibatch accuracy: 99.2%
    Validation accuracy: 79.2%
    Minibatch loss at step 2000: 0.282700
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.5%
    Minibatch loss at step 2500: 0.281390
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.5%
    Minibatch loss at step 3000: 0.277934
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.5%
    Minibatch loss at step 3500: 0.268778
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.5%
    Minibatch loss at step 4000: 0.268410
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.3%
    Minibatch loss at step 4500: 0.265331
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.4%
    Minibatch loss at step 5000: 0.263989
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.4%
    Minibatch loss at step 5500: 0.263227
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.4%
    Minibatch loss at step 6000: 0.257293
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.4%
    Minibatch loss at step 6500: 0.255185
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.4%
    Minibatch loss at step 7000: 0.256734
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.4%
    Minibatch loss at step 7500: 0.252498
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.1%
    Minibatch loss at step 8000: 0.251201
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.1%
    Minibatch loss at step 8500: 0.255578
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.2%
    Minibatch loss at step 9000: 0.256471
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.4%
    Minibatch loss at step 9500: 0.262707
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.5%
    Minibatch loss at step 10000: 0.262351
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.6%
    Test accuracy: 86.7%
    

## Problem 3

Introduce Dropout on the hidden layer of the neural network. Remember: Dropout should only be introduced during training, not evaluation, otherwise your evaluation results would be stochastic as well. TensorFlow provides nn.dropout() for that, but you have to make sure it's only inserted during training.

What happens to our extreme overfitting case?


```python
num_nodes = 1024
batch_size = 128
beta = 0.01

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  weights_1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_nodes]))
  biases_1 = tf.Variable(tf.zeros([num_nodes]))
  weights_2 = tf.Variable(
    tf.truncated_normal([num_nodes, num_labels]))
  biases_2 = tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.
  relu_layer = tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)
  # Drop out in relu layer
  keep_prob = tf.placeholder("float")
  relu_layer_dropout = tf.nn.dropout(relu_layer, keep_prob)
  logits = tf.matmul(relu_layer_dropout, weights_2) + biases_2
  
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  # L2 Regularization
  regularizer = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2)
  # loss after using L2 Regularization
  loss = tf.reduce_mean(loss + beta * regularizer)
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights_1) + biases_1), weights_2) + biases_2)
  test_prediction = tf.nn.softmax(
      tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1), weights_2) + biases_2)
```


```python
num_steps = 10001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, 
                keep_prob : 0.5}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
```

    Initialized
    Minibatch loss at step 0: 3587.162598
    Minibatch accuracy: 16.4%
    Validation accuracy: 23.3%
    Minibatch loss at step 500: 21.337776
    Minibatch accuracy: 84.4%
    Validation accuracy: 83.4%
    Minibatch loss at step 1000: 1.044617
    Minibatch accuracy: 78.9%
    Validation accuracy: 82.6%
    Minibatch loss at step 1500: 0.652918
    Minibatch accuracy: 86.7%
    Validation accuracy: 83.0%
    Minibatch loss at step 2000: 0.675978
    Minibatch accuracy: 88.3%
    Validation accuracy: 83.2%
    Minibatch loss at step 2500: 0.837159
    Minibatch accuracy: 81.2%
    Validation accuracy: 82.8%
    Minibatch loss at step 3000: 0.843557
    Minibatch accuracy: 82.8%
    Validation accuracy: 83.0%
    Minibatch loss at step 3500: 0.830758
    Minibatch accuracy: 80.5%
    Validation accuracy: 83.2%
    Minibatch loss at step 4000: 0.752345
    Minibatch accuracy: 85.2%
    Validation accuracy: 83.2%
    Minibatch loss at step 4500: 0.750211
    Minibatch accuracy: 82.0%
    Validation accuracy: 83.1%
    Minibatch loss at step 5000: 0.734512
    Minibatch accuracy: 84.4%
    Validation accuracy: 83.2%
    Minibatch loss at step 5500: 0.885577
    Minibatch accuracy: 77.3%
    Validation accuracy: 82.5%
    Minibatch loss at step 6000: 0.867540
    Minibatch accuracy: 75.8%
    Validation accuracy: 82.6%
    Minibatch loss at step 6500: 0.692108
    Minibatch accuracy: 84.4%
    Validation accuracy: 83.3%
    Minibatch loss at step 7000: 0.874965
    Minibatch accuracy: 78.1%
    Validation accuracy: 83.2%
    Minibatch loss at step 7500: 0.955431
    Minibatch accuracy: 76.6%
    Validation accuracy: 83.3%
    Minibatch loss at step 8000: 1.011686
    Minibatch accuracy: 75.0%
    Validation accuracy: 82.5%
    Minibatch loss at step 8500: 0.711539
    Minibatch accuracy: 84.4%
    Validation accuracy: 82.7%
    Minibatch loss at step 9000: 0.904487
    Minibatch accuracy: 79.7%
    Validation accuracy: 82.1%
    Minibatch loss at step 9500: 0.828240
    Minibatch accuracy: 83.6%
    Validation accuracy: 83.1%
    Minibatch loss at step 10000: 0.884038
    Minibatch accuracy: 78.1%
    Validation accuracy: 82.8%
    Test accuracy: 89.8%
    

##### Extreme Overfitting


```python
num_steps = 10001

train_dataset_2 = train_dataset[:500,:]
train_labels_2 = train_labels[:500]

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels_2.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset_2[offset:(offset + batch_size), :]
    batch_labels = train_labels_2[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels,
                keep_prob : 0.5}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
```

    Initialized
    Minibatch loss at step 0: 3556.819336
    Minibatch accuracy: 8.6%
    Validation accuracy: 27.7%
    Minibatch loss at step 500: 21.111876
    Minibatch accuracy: 100.0%
    Validation accuracy: 78.7%
    Minibatch loss at step 1000: 0.517459
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.7%
    Minibatch loss at step 1500: 0.335398
    Minibatch accuracy: 99.2%
    Validation accuracy: 79.5%
    Minibatch loss at step 2000: 0.311299
    Minibatch accuracy: 99.2%
    Validation accuracy: 79.6%
    Minibatch loss at step 2500: 0.301840
    Minibatch accuracy: 99.2%
    Validation accuracy: 80.0%
    Minibatch loss at step 3000: 0.296899
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.8%
    Minibatch loss at step 3500: 0.301349
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.5%
    Minibatch loss at step 4000: 0.296846
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.5%
    Minibatch loss at step 4500: 0.286273
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.7%
    Minibatch loss at step 5000: 0.286915
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.5%
    Minibatch loss at step 5500: 0.291484
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.7%
    Minibatch loss at step 6000: 0.274928
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.5%
    Minibatch loss at step 6500: 0.275068
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.7%
    Minibatch loss at step 7000: 0.283895
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.5%
    Minibatch loss at step 7500: 0.269533
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.4%
    Minibatch loss at step 8000: 0.270013
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.2%
    Minibatch loss at step 8500: 0.280784
    Minibatch accuracy: 99.2%
    Validation accuracy: 79.4%
    Minibatch loss at step 9000: 0.273100
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.6%
    Minibatch loss at step 9500: 0.292555
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.7%
    Minibatch loss at step 10000: 0.293184
    Minibatch accuracy: 100.0%
    Validation accuracy: 79.5%
    Test accuracy: 86.7%
    

## Problem 4

Try to get the best performance you can using a multi-layer model! The best reported test accuracy using a deep network is 97.1%.

One avenue you can explore is to add multiple layers.

Another one is to use learning rate decay:

`global_step = tf.Variable(0)  # count the number of steps taken.
learning_rate = tf.train.exponential_decay(0.5, global_step, ...)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)`



##### NN Model with 5 Hidden Layers 

* Model with 5 hidden layers
    - RELUs
    - Number of nodes in each hidden layer is 50% of that in the previous one
* Overfitting measures
    - L2 Regularization
        - Learning rate with exponential decay; starting value = 0.01
    - Dropout
* Number of steps : 15,000 



```python
hidden_1 = 1024
hidden_2 = int(hidden_1 * 0.5)
hidden_3 = int(hidden_2 * 0.5)
hidden_4 = int(hidden_3 * 0.5)
hidden_5 = int(hidden_4 * 0.5)

batch_size = 128
beta = 0.001

import math

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  # Adjusted the stddev value as per https://discussions.udacity.com/t/assignment-2-2-hidden-layers-error/183933/4 
  # Note: any stddev that doesn't produce NAN is fine.
    
  # Hidden RELU Layer 1
  weights_1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, hidden_1], 
                        stddev=2.0 / math.sqrt(float(image_size * image_size + hidden_1))))
  biases_1 = tf.Variable(tf.zeros([hidden_1]))
    
  # Hidden RELU Layer 2
  weights_2 = tf.Variable(
    tf.truncated_normal([hidden_1, hidden_2], 
                        stddev=2.0 / math.sqrt(float(hidden_1 + hidden_2))))
  biases_2 = tf.Variable(tf.zeros([hidden_2]))
  
  # Hidden RELU Layer 3
  weights_3 = tf.Variable(
    tf.truncated_normal([hidden_2, hidden_3], 
                        stddev=2.0 / math.sqrt(float(hidden_2 + hidden_3))))
  biases_3 = tf.Variable(tf.zeros([hidden_3]))
   
  # Hidden RELU Layer 4
  weights_4 = tf.Variable(
    tf.truncated_normal([hidden_3, hidden_4], 
                        stddev=2.0 / math.sqrt(float(hidden_3 + hidden_4))))
  biases_4 = tf.Variable(tf.zeros([hidden_4]))
 
    # Hidden RELU Layer 5
  weights_5 = tf.Variable(
    tf.truncated_normal([hidden_4, hidden_5], 
                        stddev=2.0 / math.sqrt(float(hidden_4 + hidden_5))))
  biases_5 = tf.Variable(tf.zeros([hidden_5]))
  
  # Outer Layer
  weights_6 = tf.Variable(
    tf.truncated_normal([hidden_5, num_labels], 
                        stddev=2.0 / math.sqrt(float(hidden_5 + num_labels))))
  biases_6= tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.
  # Hidden RELU Layer 1
  relu_layer_1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)
  # Drop out in relu layer 1
  keep_prob = tf.placeholder("float")
  relu_layer_dropout_1 = tf.nn.dropout(relu_layer_1, keep_prob)
    
  # Hidden RELU Layer 2
  relu_layer_2 = tf.nn.relu(tf.matmul(relu_layer_dropout_1, weights_2) + biases_2)
  # Drop out in relu layer 2
  relu_layer_dropout_2 = tf.nn.dropout(relu_layer_2, keep_prob)
  
  # Hidden RELU Layer 3
  relu_layer_3 = tf.nn.relu(tf.matmul(relu_layer_dropout_2, weights_3) + biases_3)
  # Drop out in relu layer 3
  relu_layer_dropout_3 = tf.nn.dropout(relu_layer_3, keep_prob)
  
  # Hidden RELU Layer 4
  relu_layer_4 = tf.nn.relu(tf.matmul(relu_layer_dropout_3, weights_4) + biases_4)
  # Drop out in relu layer 4
  relu_layer_dropout_4 = tf.nn.dropout(relu_layer_4, keep_prob)
 
  # Hidden RELU Layer 5
  relu_layer_5 = tf.nn.relu(tf.matmul(relu_layer_dropout_4, weights_5) + biases_5)
  # Drop out in relu layer 5
  relu_layer_dropout_5= tf.nn.dropout(relu_layer_5, keep_prob)
 
  # Outer Layer
  logits = tf.matmul(relu_layer_dropout_5, weights_6) + biases_6
  
  # Normal loss function
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  # L2 Regularization
  regularizer = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(
      weights_3) + tf.nn.l2_loss(weights_4) + tf.nn.l2_loss(weights_5) + tf.nn.l2_loss(weights_6)
  # loss after using L2 Regularization
  loss = tf.reduce_mean(loss + beta * regularizer)
    
  # Optimizer.
  # Decaying learning rate
  # Decay step = no of steps after which learning rate will be updated
  # Set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  global_step = tf.Variable(0)  # count the number of steps taken.
  start_learning_rate = 0.01
  learning_rate = tf.train.exponential_decay(start_learning_rate, # Base learning rate.
                                      global_step,  # global_step: Current index into the dataset. 
                                      5000,         # Decay step: each epoch here is 200000/128=1562.5 steps
                                      0.96,         # Decay rate.
                                      staircase=True)
  # Note the global_step=global_step parameter to minimize. 
  # That tells the optimizer to helpfully increment the 'global_step' parameter for you every time it trains.  
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, 
                                                                        global_step=global_step)
   
  # Predictions for the training
  train_prediction = tf.nn.softmax(logits)
  
  # Predictions for the validation data.
  valid_logits_1 = tf.matmul(tf_valid_dataset, weights_1) + biases_1
  valid_relu_1 = tf.nn.relu(valid_logits_1)
  
  valid_logits_2 = tf.matmul(valid_relu_1, weights_2) + biases_2
  valid_relu_2 = tf.nn.relu(valid_logits_2)  

  valid_logits_3 = tf.matmul(valid_relu_2, weights_3) + biases_3
  valid_relu_3 = tf.nn.relu(valid_logits_3)  
  
  valid_logits_4 = tf.matmul(valid_relu_3, weights_4) + biases_4
  valid_relu_4 = tf.nn.relu(valid_logits_4)  
  
  valid_logits_5 = tf.matmul(valid_relu_4, weights_5) + biases_5
  valid_relu_5 = tf.nn.relu(valid_logits_5)  
  
  valid_logits_6 = tf.matmul(valid_relu_5, weights_6) + biases_6
    
  valid_prediction = tf.nn.softmax(valid_logits_6)

  # Predictions for the test data.
  test_logits_1 = tf.matmul(tf_test_dataset, weights_1) + biases_1
  test_relu_1 = tf.nn.relu(test_logits_1)
  
  test_logits_2 = tf.matmul(test_relu_1, weights_2) + biases_2
  test_relu_2 = tf.nn.relu(test_logits_2)  

  test_logits_3 = tf.matmul(test_relu_2, weights_3) + biases_3
  test_relu_3 = tf.nn.relu(test_logits_3)  

  test_logits_4 = tf.matmul(test_relu_3, weights_4) + biases_4
  test_relu_4 = tf.nn.relu(test_logits_4)  

  test_logits_5 = tf.matmul(test_relu_4, weights_5) + biases_5
  test_relu_5 = tf.nn.relu(test_logits_5)  

  test_logits_6 = tf.matmul(test_relu_5, weights_6) + biases_6  
  test_prediction = tf.nn.softmax(test_logits_6)
```


```python
num_steps = 150001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, 
                keep_prob : 0.5}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
```

    Initialized
    Minibatch loss at step 0: 6.804766
    Minibatch accuracy: 10.2%
    Validation accuracy: 9.3%
    Minibatch loss at step 500: 4.032598
    Minibatch accuracy: 10.2%
    Validation accuracy: 23.6%
    Minibatch loss at step 1000: 3.750102
    Minibatch accuracy: 25.8%
    Validation accuracy: 42.3%
    Minibatch loss at step 1500: 3.245124
    Minibatch accuracy: 38.3%
    Validation accuracy: 58.0%
    Minibatch loss at step 2000: 3.210968
    Minibatch accuracy: 43.0%
    Validation accuracy: 67.2%
    Minibatch loss at step 2500: 3.032999
    Minibatch accuracy: 53.9%
    Validation accuracy: 71.9%
    Minibatch loss at step 3000: 2.735332
    Minibatch accuracy: 63.3%
    Validation accuracy: 73.2%
    Minibatch loss at step 3500: 2.756519
    Minibatch accuracy: 59.4%
    Validation accuracy: 78.4%
    Minibatch loss at step 4000: 2.788174
    Minibatch accuracy: 68.8%
    Validation accuracy: 79.8%
    Minibatch loss at step 4500: 2.515474
    Minibatch accuracy: 72.7%
    Validation accuracy: 80.7%
    Minibatch loss at step 5000: 2.433222
    Minibatch accuracy: 70.3%
    Validation accuracy: 81.2%
    Minibatch loss at step 5500: 2.423468
    Minibatch accuracy: 68.0%
    Validation accuracy: 81.7%
    Minibatch loss at step 6000: 2.527008
    Minibatch accuracy: 68.8%
    Validation accuracy: 81.9%
    Minibatch loss at step 6500: 2.194419
    Minibatch accuracy: 75.8%
    Validation accuracy: 82.3%
    Minibatch loss at step 7000: 2.306954
    Minibatch accuracy: 72.7%
    Validation accuracy: 82.6%
    Minibatch loss at step 7500: 2.527934
    Minibatch accuracy: 75.0%
    Validation accuracy: 82.8%
    Minibatch loss at step 8000: 2.372506
    Minibatch accuracy: 74.2%
    Validation accuracy: 83.0%
    Minibatch loss at step 8500: 1.986309
    Minibatch accuracy: 82.0%
    Validation accuracy: 83.2%
    Minibatch loss at step 9000: 2.191247
    Minibatch accuracy: 79.7%
    Validation accuracy: 83.2%
    Minibatch loss at step 9500: 2.206452
    Minibatch accuracy: 78.1%
    Validation accuracy: 83.4%
    Minibatch loss at step 10000: 2.139791
    Minibatch accuracy: 79.7%
    Validation accuracy: 83.3%
    Minibatch loss at step 10500: 2.114667
    Minibatch accuracy: 81.2%
    Validation accuracy: 83.7%
    Minibatch loss at step 11000: 2.124454
    Minibatch accuracy: 82.8%
    Validation accuracy: 83.7%
    Minibatch loss at step 11500: 2.070670
    Minibatch accuracy: 79.7%
    Validation accuracy: 83.8%
    Minibatch loss at step 12000: 2.280128
    Minibatch accuracy: 72.7%
    Validation accuracy: 83.9%
    Minibatch loss at step 12500: 2.022823
    Minibatch accuracy: 80.5%
    Validation accuracy: 84.1%
    Minibatch loss at step 13000: 2.119970
    Minibatch accuracy: 77.3%
    Validation accuracy: 84.0%
    Minibatch loss at step 13500: 2.081277
    Minibatch accuracy: 78.1%
    Validation accuracy: 84.1%
    Minibatch loss at step 14000: 1.943875
    Minibatch accuracy: 81.2%
    Validation accuracy: 84.2%
    Minibatch loss at step 14500: 1.997872
    Minibatch accuracy: 78.1%
    Validation accuracy: 84.4%
    Minibatch loss at step 15000: 1.845917
    Minibatch accuracy: 84.4%
    Validation accuracy: 84.5%
    Minibatch loss at step 15500: 1.956824
    Minibatch accuracy: 78.9%
    Validation accuracy: 84.6%
    Minibatch loss at step 16000: 1.840317
    Minibatch accuracy: 84.4%
    Validation accuracy: 84.7%
    Minibatch loss at step 16500: 1.730737
    Minibatch accuracy: 86.7%
    Validation accuracy: 84.8%
    Minibatch loss at step 17000: 1.782844
    Minibatch accuracy: 84.4%
    Validation accuracy: 84.9%
    Minibatch loss at step 17500: 1.551369
    Minibatch accuracy: 91.4%
    Validation accuracy: 84.9%
    Minibatch loss at step 18000: 1.747159
    Minibatch accuracy: 82.8%
    Validation accuracy: 84.9%
    Minibatch loss at step 18500: 1.822943
    Minibatch accuracy: 84.4%
    Validation accuracy: 85.0%
    Minibatch loss at step 19000: 1.577337
    Minibatch accuracy: 88.3%
    Validation accuracy: 85.4%
    Minibatch loss at step 19500: 1.810753
    Minibatch accuracy: 83.6%
    Validation accuracy: 85.4%
    Minibatch loss at step 20000: 1.903370
    Minibatch accuracy: 77.3%
    Validation accuracy: 85.4%
    Minibatch loss at step 20500: 1.624809
    Minibatch accuracy: 86.7%
    Validation accuracy: 85.5%
    Minibatch loss at step 21000: 1.806881
    Minibatch accuracy: 81.2%
    Validation accuracy: 85.8%
    Minibatch loss at step 21500: 1.738962
    Minibatch accuracy: 82.0%
    Validation accuracy: 85.6%
    Minibatch loss at step 22000: 1.577854
    Minibatch accuracy: 85.9%
    Validation accuracy: 85.7%
    Minibatch loss at step 22500: 1.641550
    Minibatch accuracy: 83.6%
    Validation accuracy: 85.8%
    Minibatch loss at step 23000: 1.692333
    Minibatch accuracy: 82.8%
    Validation accuracy: 85.8%
    Minibatch loss at step 23500: 1.482873
    Minibatch accuracy: 91.4%
    Validation accuracy: 86.0%
    Minibatch loss at step 24000: 1.685335
    Minibatch accuracy: 82.0%
    Validation accuracy: 86.0%
    Minibatch loss at step 24500: 1.671486
    Minibatch accuracy: 81.2%
    Validation accuracy: 86.0%
    Minibatch loss at step 25000: 1.783657
    Minibatch accuracy: 82.8%
    Validation accuracy: 86.0%
    Minibatch loss at step 25500: 1.589290
    Minibatch accuracy: 83.6%
    Validation accuracy: 86.1%
    Minibatch loss at step 26000: 1.636632
    Minibatch accuracy: 81.2%
    Validation accuracy: 86.2%
    Minibatch loss at step 26500: 1.525098
    Minibatch accuracy: 87.5%
    Validation accuracy: 86.3%
    Minibatch loss at step 27000: 1.561834
    Minibatch accuracy: 84.4%
    Validation accuracy: 86.3%
    Minibatch loss at step 27500: 1.757689
    Minibatch accuracy: 82.0%
    Validation accuracy: 86.5%
    Minibatch loss at step 28000: 1.485148
    Minibatch accuracy: 84.4%
    Validation accuracy: 86.5%
    Minibatch loss at step 28500: 1.483319
    Minibatch accuracy: 85.9%
    Validation accuracy: 86.4%
    Minibatch loss at step 29000: 1.785676
    Minibatch accuracy: 81.2%
    Validation accuracy: 86.5%
    Minibatch loss at step 29500: 1.631732
    Minibatch accuracy: 84.4%
    Validation accuracy: 86.6%
    Minibatch loss at step 30000: 1.524098
    Minibatch accuracy: 82.8%
    Validation accuracy: 86.6%
    Minibatch loss at step 30500: 1.453226
    Minibatch accuracy: 86.7%
    Validation accuracy: 86.7%
    Minibatch loss at step 31000: 1.572179
    Minibatch accuracy: 82.8%
    Validation accuracy: 86.8%
    Minibatch loss at step 31500: 1.488299
    Minibatch accuracy: 84.4%
    Validation accuracy: 86.7%
    Minibatch loss at step 32000: 1.387465
    Minibatch accuracy: 85.2%
    Validation accuracy: 86.8%
    Minibatch loss at step 32500: 1.548264
    Minibatch accuracy: 82.8%
    Validation accuracy: 86.8%
    Minibatch loss at step 33000: 1.546062
    Minibatch accuracy: 82.8%
    Validation accuracy: 86.8%
    Minibatch loss at step 33500: 1.295459
    Minibatch accuracy: 88.3%
    Validation accuracy: 86.9%
    Minibatch loss at step 34000: 1.754502
    Minibatch accuracy: 75.0%
    Validation accuracy: 86.9%
    Minibatch loss at step 34500: 1.656559
    Minibatch accuracy: 80.5%
    Validation accuracy: 87.0%
    Minibatch loss at step 35000: 1.339179
    Minibatch accuracy: 91.4%
    Validation accuracy: 87.2%
    Minibatch loss at step 35500: 1.502674
    Minibatch accuracy: 84.4%
    Validation accuracy: 87.0%
    Minibatch loss at step 36000: 1.497382
    Minibatch accuracy: 82.0%
    Validation accuracy: 87.0%
    Minibatch loss at step 36500: 1.365980
    Minibatch accuracy: 86.7%
    Validation accuracy: 87.1%
    Minibatch loss at step 37000: 1.418648
    Minibatch accuracy: 82.8%
    Validation accuracy: 87.0%
    Minibatch loss at step 37500: 1.158665
    Minibatch accuracy: 91.4%
    Validation accuracy: 87.0%
    Minibatch loss at step 38000: 1.470743
    Minibatch accuracy: 82.8%
    Validation accuracy: 87.2%
    Minibatch loss at step 38500: 1.363068
    Minibatch accuracy: 85.2%
    Validation accuracy: 87.4%
    Minibatch loss at step 39000: 1.560978
    Minibatch accuracy: 81.2%
    Validation accuracy: 87.3%
    Minibatch loss at step 39500: 1.455091
    Minibatch accuracy: 78.9%
    Validation accuracy: 87.4%
    Minibatch loss at step 40000: 1.297829
    Minibatch accuracy: 88.3%
    Validation accuracy: 87.3%
    Minibatch loss at step 40500: 1.431495
    Minibatch accuracy: 83.6%
    Validation accuracy: 87.4%
    Minibatch loss at step 41000: 1.227817
    Minibatch accuracy: 89.8%
    Validation accuracy: 87.4%
    Minibatch loss at step 41500: 1.489477
    Minibatch accuracy: 85.2%
    Validation accuracy: 87.4%
    Minibatch loss at step 42000: 1.267743
    Minibatch accuracy: 91.4%
    Validation accuracy: 87.5%
    Minibatch loss at step 42500: 1.322699
    Minibatch accuracy: 85.9%
    Validation accuracy: 87.5%
    Minibatch loss at step 43000: 1.301431
    Minibatch accuracy: 85.2%
    Validation accuracy: 87.5%
    Minibatch loss at step 43500: 1.302545
    Minibatch accuracy: 85.9%
    Validation accuracy: 87.5%
    Minibatch loss at step 44000: 1.343225
    Minibatch accuracy: 85.9%
    Validation accuracy: 87.6%
    Minibatch loss at step 44500: 1.286568
    Minibatch accuracy: 86.7%
    Validation accuracy: 87.5%
    Minibatch loss at step 45000: 1.310206
    Minibatch accuracy: 89.1%
    Validation accuracy: 87.7%
    Minibatch loss at step 45500: 1.165762
    Minibatch accuracy: 87.5%
    Validation accuracy: 87.6%
    Minibatch loss at step 46000: 1.145842
    Minibatch accuracy: 92.2%
    Validation accuracy: 87.6%
    Minibatch loss at step 46500: 1.536286
    Minibatch accuracy: 77.3%
    Validation accuracy: 87.6%
    Minibatch loss at step 47000: 1.176394
    Minibatch accuracy: 89.1%
    Validation accuracy: 87.7%
    Minibatch loss at step 47500: 1.180098
    Minibatch accuracy: 87.5%
    Validation accuracy: 87.7%
    Minibatch loss at step 48000: 1.207114
    Minibatch accuracy: 86.7%
    Validation accuracy: 87.7%
    Minibatch loss at step 48500: 1.258222
    Minibatch accuracy: 84.4%
    Validation accuracy: 87.8%
    Minibatch loss at step 49000: 1.286874
    Minibatch accuracy: 85.2%
    Validation accuracy: 87.8%
    Minibatch loss at step 49500: 1.235683
    Minibatch accuracy: 86.7%
    Validation accuracy: 87.8%
    Minibatch loss at step 50000: 1.308735
    Minibatch accuracy: 81.2%
    Validation accuracy: 87.8%
    Minibatch loss at step 50500: 1.371897
    Minibatch accuracy: 83.6%
    Validation accuracy: 87.9%
    Minibatch loss at step 51000: 1.143800
    Minibatch accuracy: 88.3%
    Validation accuracy: 87.8%
    Minibatch loss at step 51500: 1.182097
    Minibatch accuracy: 88.3%
    Validation accuracy: 87.9%
    Minibatch loss at step 52000: 1.285549
    Minibatch accuracy: 89.8%
    Validation accuracy: 88.0%
    Minibatch loss at step 52500: 1.125270
    Minibatch accuracy: 87.5%
    Validation accuracy: 88.0%
    Minibatch loss at step 53000: 1.069888
    Minibatch accuracy: 91.4%
    Validation accuracy: 88.0%
    Minibatch loss at step 53500: 1.324988
    Minibatch accuracy: 82.0%
    Validation accuracy: 88.2%
    Minibatch loss at step 54000: 1.287739
    Minibatch accuracy: 82.8%
    Validation accuracy: 87.9%
    Minibatch loss at step 54500: 1.337905
    Minibatch accuracy: 84.4%
    Validation accuracy: 88.0%
    Minibatch loss at step 55000: 1.101998
    Minibatch accuracy: 88.3%
    Validation accuracy: 88.1%
    Minibatch loss at step 55500: 1.287743
    Minibatch accuracy: 82.0%
    Validation accuracy: 88.1%
    Minibatch loss at step 56000: 1.122588
    Minibatch accuracy: 85.9%
    Validation accuracy: 88.2%
    Minibatch loss at step 56500: 1.028130
    Minibatch accuracy: 93.0%
    Validation accuracy: 88.2%
    Minibatch loss at step 57000: 1.072535
    Minibatch accuracy: 89.1%
    Validation accuracy: 88.0%
    Minibatch loss at step 57500: 1.105749
    Minibatch accuracy: 86.7%
    Validation accuracy: 88.2%
    Minibatch loss at step 58000: 0.958441
    Minibatch accuracy: 93.0%
    Validation accuracy: 88.1%
    Minibatch loss at step 58500: 1.017952
    Minibatch accuracy: 89.8%
    Validation accuracy: 88.2%
    Minibatch loss at step 59000: 1.227513
    Minibatch accuracy: 82.8%
    Validation accuracy: 88.2%
    Minibatch loss at step 59500: 1.037941
    Minibatch accuracy: 89.1%
    Validation accuracy: 88.4%
    Minibatch loss at step 60000: 1.110142
    Minibatch accuracy: 88.3%
    Validation accuracy: 88.2%
    Minibatch loss at step 60500: 1.115096
    Minibatch accuracy: 89.8%
    Validation accuracy: 88.2%
    Minibatch loss at step 61000: 0.944245
    Minibatch accuracy: 92.2%
    Validation accuracy: 88.3%
    Minibatch loss at step 61500: 1.039468
    Minibatch accuracy: 89.1%
    Validation accuracy: 88.3%
    Minibatch loss at step 62000: 0.910933
    Minibatch accuracy: 94.5%
    Validation accuracy: 88.3%
    Minibatch loss at step 62500: 1.207104
    Minibatch accuracy: 79.7%
    Validation accuracy: 88.4%
    Minibatch loss at step 63000: 1.136097
    Minibatch accuracy: 86.7%
    Validation accuracy: 88.5%
    Minibatch loss at step 63500: 1.048399
    Minibatch accuracy: 85.9%
    Validation accuracy: 88.4%
    Minibatch loss at step 64000: 1.040099
    Minibatch accuracy: 87.5%
    Validation accuracy: 88.6%
    Minibatch loss at step 64500: 1.223176
    Minibatch accuracy: 84.4%
    Validation accuracy: 88.5%
    Minibatch loss at step 65000: 1.026851
    Minibatch accuracy: 88.3%
    Validation accuracy: 88.5%
    Minibatch loss at step 65500: 0.997451
    Minibatch accuracy: 91.4%
    Validation accuracy: 88.5%
    Minibatch loss at step 66000: 1.038661
    Minibatch accuracy: 85.9%
    Validation accuracy: 88.5%
    Minibatch loss at step 66500: 1.180174
    Minibatch accuracy: 86.7%
    Validation accuracy: 88.5%
    Minibatch loss at step 67000: 1.021461
    Minibatch accuracy: 88.3%
    Validation accuracy: 88.6%
    Minibatch loss at step 67500: 0.918733
    Minibatch accuracy: 92.2%
    Validation accuracy: 88.5%
    Minibatch loss at step 68000: 1.084973
    Minibatch accuracy: 87.5%
    Validation accuracy: 88.5%
    Minibatch loss at step 68500: 0.938965
    Minibatch accuracy: 92.2%
    Validation accuracy: 88.6%
    Minibatch loss at step 69000: 0.974708
    Minibatch accuracy: 88.3%
    Validation accuracy: 88.5%
    Minibatch loss at step 69500: 1.025112
    Minibatch accuracy: 87.5%
    Validation accuracy: 88.5%
    Minibatch loss at step 70000: 1.090736
    Minibatch accuracy: 85.2%
    Validation accuracy: 88.5%
    Minibatch loss at step 70500: 0.978542
    Minibatch accuracy: 87.5%
    Validation accuracy: 88.6%
    Minibatch loss at step 71000: 1.105583
    Minibatch accuracy: 85.9%
    Validation accuracy: 88.7%
    Minibatch loss at step 71500: 1.014713
    Minibatch accuracy: 85.9%
    Validation accuracy: 88.6%
    Minibatch loss at step 72000: 0.954982
    Minibatch accuracy: 89.1%
    Validation accuracy: 88.6%
    Minibatch loss at step 72500: 0.996724
    Minibatch accuracy: 89.1%
    Validation accuracy: 88.7%
    Minibatch loss at step 73000: 1.063256
    Minibatch accuracy: 85.9%
    Validation accuracy: 88.5%
    Minibatch loss at step 73500: 1.104174
    Minibatch accuracy: 83.6%
    Validation accuracy: 88.5%
    Minibatch loss at step 74000: 1.052952
    Minibatch accuracy: 87.5%
    Validation accuracy: 88.6%
    Minibatch loss at step 74500: 0.833315
    Minibatch accuracy: 93.8%
    Validation accuracy: 88.6%
    Minibatch loss at step 75000: 1.003028
    Minibatch accuracy: 88.3%
    Validation accuracy: 88.6%
    Minibatch loss at step 75500: 0.942692
    Minibatch accuracy: 92.2%
    Validation accuracy: 88.6%
    Minibatch loss at step 76000: 0.908086
    Minibatch accuracy: 93.8%
    Validation accuracy: 88.6%
    Minibatch loss at step 76500: 0.901587
    Minibatch accuracy: 89.8%
    Validation accuracy: 88.8%
    Minibatch loss at step 77000: 1.083122
    Minibatch accuracy: 82.8%
    Validation accuracy: 88.7%
    Minibatch loss at step 77500: 0.960811
    Minibatch accuracy: 89.8%
    Validation accuracy: 88.6%
    Minibatch loss at step 78000: 0.892756
    Minibatch accuracy: 89.1%
    Validation accuracy: 88.7%
    Minibatch loss at step 78500: 1.080168
    Minibatch accuracy: 86.7%
    Validation accuracy: 88.7%
    Minibatch loss at step 79000: 0.849935
    Minibatch accuracy: 89.8%
    Validation accuracy: 88.7%
    Minibatch loss at step 79500: 1.000796
    Minibatch accuracy: 85.2%
    Validation accuracy: 88.8%
    Minibatch loss at step 80000: 0.847333
    Minibatch accuracy: 90.6%
    Validation accuracy: 88.8%
    Minibatch loss at step 80500: 0.967551
    Minibatch accuracy: 89.8%
    Validation accuracy: 88.9%
    Minibatch loss at step 81000: 0.914809
    Minibatch accuracy: 89.1%
    Validation accuracy: 88.8%
    Minibatch loss at step 81500: 0.987431
    Minibatch accuracy: 88.3%
    Validation accuracy: 88.8%
    Minibatch loss at step 82000: 0.864907
    Minibatch accuracy: 90.6%
    Validation accuracy: 88.8%
    Minibatch loss at step 82500: 0.962669
    Minibatch accuracy: 89.8%
    Validation accuracy: 88.9%
    Minibatch loss at step 83000: 1.016966
    Minibatch accuracy: 90.6%
    Validation accuracy: 89.0%
    Minibatch loss at step 83500: 0.861801
    Minibatch accuracy: 90.6%
    Validation accuracy: 88.9%
    Minibatch loss at step 84000: 0.960644
    Minibatch accuracy: 88.3%
    Validation accuracy: 88.9%
    Minibatch loss at step 84500: 0.844911
    Minibatch accuracy: 89.8%
    Validation accuracy: 88.9%
    Minibatch loss at step 85000: 0.910592
    Minibatch accuracy: 91.4%
    Validation accuracy: 88.9%
    Minibatch loss at step 85500: 0.998953
    Minibatch accuracy: 84.4%
    Validation accuracy: 89.0%
    Minibatch loss at step 86000: 0.871746
    Minibatch accuracy: 91.4%
    Validation accuracy: 89.0%
    Minibatch loss at step 86500: 0.940663
    Minibatch accuracy: 90.6%
    Validation accuracy: 89.1%
    Minibatch loss at step 87000: 0.793929
    Minibatch accuracy: 92.2%
    Validation accuracy: 89.0%
    Minibatch loss at step 87500: 0.923824
    Minibatch accuracy: 86.7%
    Validation accuracy: 89.0%
    Minibatch loss at step 88000: 0.846314
    Minibatch accuracy: 88.3%
    Validation accuracy: 89.0%
    Minibatch loss at step 88500: 0.788783
    Minibatch accuracy: 91.4%
    Validation accuracy: 89.0%
    Minibatch loss at step 89000: 1.070700
    Minibatch accuracy: 84.4%
    Validation accuracy: 89.1%
    Minibatch loss at step 89500: 0.955315
    Minibatch accuracy: 88.3%
    Validation accuracy: 89.1%
    Minibatch loss at step 90000: 0.866157
    Minibatch accuracy: 87.5%
    Validation accuracy: 89.1%
    Minibatch loss at step 90500: 0.790577
    Minibatch accuracy: 91.4%
    Validation accuracy: 89.2%
    Minibatch loss at step 91000: 0.904507
    Minibatch accuracy: 90.6%
    Validation accuracy: 89.1%
    Minibatch loss at step 91500: 0.927198
    Minibatch accuracy: 88.3%
    Validation accuracy: 89.2%
    Minibatch loss at step 92000: 0.851750
    Minibatch accuracy: 90.6%
    Validation accuracy: 89.0%
    Minibatch loss at step 92500: 0.868930
    Minibatch accuracy: 89.8%
    Validation accuracy: 89.0%
    Minibatch loss at step 93000: 0.916286
    Minibatch accuracy: 85.9%
    Validation accuracy: 89.2%
    Minibatch loss at step 93500: 0.846813
    Minibatch accuracy: 89.8%
    Validation accuracy: 89.2%
    Minibatch loss at step 94000: 0.904889
    Minibatch accuracy: 89.8%
    Validation accuracy: 89.2%
    Minibatch loss at step 94500: 0.936953
    Minibatch accuracy: 86.7%
    Validation accuracy: 89.1%
    Minibatch loss at step 95000: 0.886565
    Minibatch accuracy: 87.5%
    Validation accuracy: 89.2%
    Minibatch loss at step 95500: 1.068510
    Minibatch accuracy: 81.2%
    Validation accuracy: 89.1%
    Minibatch loss at step 96000: 1.066453
    Minibatch accuracy: 84.4%
    Validation accuracy: 89.3%
    Minibatch loss at step 96500: 0.845059
    Minibatch accuracy: 87.5%
    Validation accuracy: 89.2%
    Minibatch loss at step 97000: 0.714331
    Minibatch accuracy: 93.8%
    Validation accuracy: 89.2%
    Minibatch loss at step 97500: 0.871103
    Minibatch accuracy: 85.9%
    Validation accuracy: 89.2%
    Minibatch loss at step 98000: 0.780440
    Minibatch accuracy: 92.2%
    Validation accuracy: 89.2%
    Minibatch loss at step 98500: 0.735306
    Minibatch accuracy: 95.3%
    Validation accuracy: 89.3%
    Minibatch loss at step 99000: 0.867080
    Minibatch accuracy: 89.1%
    Validation accuracy: 89.3%
    Minibatch loss at step 99500: 0.818350
    Minibatch accuracy: 89.1%
    Validation accuracy: 89.4%
    Minibatch loss at step 100000: 0.832023
    Minibatch accuracy: 89.1%
    Validation accuracy: 89.3%
    Minibatch loss at step 100500: 0.784346
    Minibatch accuracy: 89.1%
    Validation accuracy: 89.5%
    Minibatch loss at step 101000: 1.007184
    Minibatch accuracy: 85.2%
    Validation accuracy: 89.4%
    Minibatch loss at step 101500: 0.748079
    Minibatch accuracy: 92.2%
    Validation accuracy: 89.5%
    Minibatch loss at step 102000: 0.876657
    Minibatch accuracy: 85.2%
    Validation accuracy: 89.5%
    Minibatch loss at step 102500: 0.823506
    Minibatch accuracy: 89.1%
    Validation accuracy: 89.4%
    Minibatch loss at step 103000: 0.866988
    Minibatch accuracy: 85.9%
    Validation accuracy: 89.5%
    Minibatch loss at step 103500: 0.900799
    Minibatch accuracy: 88.3%
    Validation accuracy: 89.4%
    Minibatch loss at step 104000: 0.766707
    Minibatch accuracy: 90.6%
    Validation accuracy: 89.5%
    Minibatch loss at step 104500: 0.978583
    Minibatch accuracy: 85.9%
    Validation accuracy: 89.4%
    Minibatch loss at step 105000: 0.765852
    Minibatch accuracy: 91.4%
    Validation accuracy: 89.4%
    Minibatch loss at step 105500: 0.925021
    Minibatch accuracy: 85.9%
    Validation accuracy: 89.5%
    Minibatch loss at step 106000: 0.951622
    Minibatch accuracy: 83.6%
    Validation accuracy: 89.6%
    Minibatch loss at step 106500: 0.873775
    Minibatch accuracy: 90.6%
    Validation accuracy: 89.6%
    Minibatch loss at step 107000: 0.786658
    Minibatch accuracy: 91.4%
    Validation accuracy: 89.5%
    Minibatch loss at step 107500: 0.881593
    Minibatch accuracy: 88.3%
    Validation accuracy: 89.6%
    Minibatch loss at step 108000: 0.802791
    Minibatch accuracy: 91.4%
    Validation accuracy: 89.6%
    Minibatch loss at step 108500: 0.927378
    Minibatch accuracy: 85.9%
    Validation accuracy: 89.6%
    Minibatch loss at step 109000: 0.922087
    Minibatch accuracy: 84.4%
    Validation accuracy: 89.5%
    Minibatch loss at step 109500: 0.789224
    Minibatch accuracy: 89.1%
    Validation accuracy: 89.6%
    Minibatch loss at step 110000: 0.896032
    Minibatch accuracy: 88.3%
    Validation accuracy: 89.6%
    Minibatch loss at step 110500: 0.830060
    Minibatch accuracy: 88.3%
    Validation accuracy: 89.6%
    Minibatch loss at step 111000: 1.172267
    Minibatch accuracy: 81.2%
    Validation accuracy: 89.6%
    Minibatch loss at step 111500: 0.695289
    Minibatch accuracy: 93.0%
    Validation accuracy: 89.6%
    Minibatch loss at step 112000: 0.768740
    Minibatch accuracy: 89.1%
    Validation accuracy: 89.5%
    Minibatch loss at step 112500: 0.913301
    Minibatch accuracy: 84.4%
    Validation accuracy: 89.5%
    Minibatch loss at step 113000: 0.660272
    Minibatch accuracy: 93.8%
    Validation accuracy: 89.6%
    Minibatch loss at step 113500: 0.892656
    Minibatch accuracy: 88.3%
    Validation accuracy: 89.6%
    Minibatch loss at step 114000: 0.762000
    Minibatch accuracy: 90.6%
    Validation accuracy: 89.6%
    Minibatch loss at step 114500: 0.945538
    Minibatch accuracy: 89.1%
    Validation accuracy: 89.6%
    Minibatch loss at step 115000: 0.768693
    Minibatch accuracy: 90.6%
    Validation accuracy: 89.5%
    Minibatch loss at step 115500: 0.959077
    Minibatch accuracy: 85.9%
    Validation accuracy: 89.7%
    Minibatch loss at step 116000: 0.725839
    Minibatch accuracy: 91.4%
    Validation accuracy: 89.7%
    Minibatch loss at step 116500: 0.938509
    Minibatch accuracy: 82.8%
    Validation accuracy: 89.7%
    Minibatch loss at step 117000: 0.930339
    Minibatch accuracy: 84.4%
    Validation accuracy: 89.8%
    Minibatch loss at step 117500: 0.728699
    Minibatch accuracy: 89.8%
    Validation accuracy: 89.8%
    Minibatch loss at step 118000: 0.832689
    Minibatch accuracy: 88.3%
    Validation accuracy: 89.8%
    Minibatch loss at step 118500: 0.767989
    Minibatch accuracy: 90.6%
    Validation accuracy: 89.7%
    Minibatch loss at step 119000: 0.737256
    Minibatch accuracy: 89.8%
    Validation accuracy: 89.8%
    Minibatch loss at step 119500: 0.760673
    Minibatch accuracy: 90.6%
    Validation accuracy: 89.7%
    Minibatch loss at step 120000: 0.848464
    Minibatch accuracy: 88.3%
    Validation accuracy: 89.8%
    Minibatch loss at step 120500: 0.820310
    Minibatch accuracy: 89.1%
    Validation accuracy: 89.8%
    Minibatch loss at step 121000: 0.768708
    Minibatch accuracy: 89.8%
    Validation accuracy: 89.7%
    Minibatch loss at step 121500: 0.817181
    Minibatch accuracy: 88.3%
    Validation accuracy: 89.8%
    Minibatch loss at step 122000: 0.787563
    Minibatch accuracy: 87.5%
    Validation accuracy: 89.9%
    Minibatch loss at step 122500: 0.800010
    Minibatch accuracy: 87.5%
    Validation accuracy: 89.8%
    Minibatch loss at step 123000: 0.722303
    Minibatch accuracy: 89.8%
    Validation accuracy: 89.7%
    Minibatch loss at step 123500: 0.680128
    Minibatch accuracy: 93.0%
    Validation accuracy: 89.9%
    Minibatch loss at step 124000: 0.823754
    Minibatch accuracy: 88.3%
    Validation accuracy: 89.8%
    Minibatch loss at step 124500: 0.821870
    Minibatch accuracy: 85.9%
    Validation accuracy: 89.8%
    Minibatch loss at step 125000: 0.869564
    Minibatch accuracy: 85.9%
    Validation accuracy: 89.8%
    Minibatch loss at step 125500: 0.695707
    Minibatch accuracy: 91.4%
    Validation accuracy: 89.9%
    Minibatch loss at step 126000: 0.664929
    Minibatch accuracy: 92.2%
    Validation accuracy: 89.9%
    Minibatch loss at step 126500: 0.809050
    Minibatch accuracy: 89.1%
    Validation accuracy: 89.9%
    Minibatch loss at step 127000: 0.895863
    Minibatch accuracy: 89.1%
    Validation accuracy: 89.8%
    Minibatch loss at step 127500: 0.919034
    Minibatch accuracy: 83.6%
    Validation accuracy: 89.9%
    Minibatch loss at step 128000: 0.840429
    Minibatch accuracy: 88.3%
    Validation accuracy: 89.9%
    Minibatch loss at step 128500: 0.839153
    Minibatch accuracy: 89.1%
    Validation accuracy: 89.9%
    Minibatch loss at step 129000: 0.836392
    Minibatch accuracy: 87.5%
    Validation accuracy: 89.9%
    Minibatch loss at step 129500: 0.790722
    Minibatch accuracy: 89.1%
    Validation accuracy: 90.0%
    Minibatch loss at step 130000: 0.687479
    Minibatch accuracy: 91.4%
    Validation accuracy: 89.9%
    Minibatch loss at step 130500: 0.815693
    Minibatch accuracy: 85.9%
    Validation accuracy: 89.9%
    Minibatch loss at step 131000: 0.627335
    Minibatch accuracy: 92.2%
    Validation accuracy: 89.9%
    Minibatch loss at step 131500: 0.733021
    Minibatch accuracy: 89.8%
    Validation accuracy: 89.9%
    Minibatch loss at step 132000: 0.713284
    Minibatch accuracy: 92.2%
    Validation accuracy: 90.0%
    Minibatch loss at step 132500: 0.794644
    Minibatch accuracy: 88.3%
    Validation accuracy: 90.0%
    Minibatch loss at step 133000: 0.714505
    Minibatch accuracy: 88.3%
    Validation accuracy: 90.0%
    Minibatch loss at step 133500: 0.788063
    Minibatch accuracy: 86.7%
    Validation accuracy: 89.9%
    Minibatch loss at step 134000: 0.872208
    Minibatch accuracy: 89.1%
    Validation accuracy: 89.9%
    Minibatch loss at step 134500: 0.929759
    Minibatch accuracy: 86.7%
    Validation accuracy: 90.0%
    Minibatch loss at step 135000: 0.882237
    Minibatch accuracy: 87.5%
    Validation accuracy: 90.0%
    Minibatch loss at step 135500: 0.718481
    Minibatch accuracy: 92.2%
    Validation accuracy: 89.9%
    Minibatch loss at step 136000: 0.911437
    Minibatch accuracy: 83.6%
    Validation accuracy: 90.0%
    Minibatch loss at step 136500: 0.726614
    Minibatch accuracy: 88.3%
    Validation accuracy: 89.9%
    Minibatch loss at step 137000: 0.759728
    Minibatch accuracy: 89.8%
    Validation accuracy: 89.9%
    Minibatch loss at step 137500: 0.773656
    Minibatch accuracy: 89.8%
    Validation accuracy: 89.9%
    Minibatch loss at step 138000: 0.713713
    Minibatch accuracy: 89.8%
    Validation accuracy: 90.0%
    Minibatch loss at step 138500: 0.744245
    Minibatch accuracy: 88.3%
    Validation accuracy: 90.0%
    Minibatch loss at step 139000: 0.784047
    Minibatch accuracy: 88.3%
    Validation accuracy: 90.0%
    Minibatch loss at step 139500: 0.758266
    Minibatch accuracy: 89.1%
    Validation accuracy: 90.0%
    Minibatch loss at step 140000: 0.651109
    Minibatch accuracy: 93.0%
    Validation accuracy: 90.0%
    Minibatch loss at step 140500: 0.743109
    Minibatch accuracy: 87.5%
    Validation accuracy: 90.0%
    Minibatch loss at step 141000: 0.768819
    Minibatch accuracy: 89.1%
    Validation accuracy: 89.9%
    Minibatch loss at step 141500: 0.940827
    Minibatch accuracy: 86.7%
    Validation accuracy: 90.0%
    Minibatch loss at step 142000: 0.820918
    Minibatch accuracy: 85.2%
    Validation accuracy: 89.9%
    Minibatch loss at step 142500: 0.774908
    Minibatch accuracy: 85.9%
    Validation accuracy: 90.0%
    Minibatch loss at step 143000: 0.759893
    Minibatch accuracy: 89.1%
    Validation accuracy: 89.9%
    Minibatch loss at step 143500: 0.912810
    Minibatch accuracy: 85.9%
    Validation accuracy: 90.1%
    Minibatch loss at step 144000: 0.706679
    Minibatch accuracy: 89.8%
    Validation accuracy: 89.9%
    Minibatch loss at step 144500: 0.814118
    Minibatch accuracy: 85.9%
    Validation accuracy: 90.0%
    Minibatch loss at step 145000: 0.724268
    Minibatch accuracy: 89.8%
    Validation accuracy: 90.0%
    Minibatch loss at step 145500: 0.601242
    Minibatch accuracy: 93.0%
    Validation accuracy: 89.9%
    Minibatch loss at step 146000: 0.834309
    Minibatch accuracy: 86.7%
    Validation accuracy: 89.9%
    Minibatch loss at step 146500: 0.674266
    Minibatch accuracy: 90.6%
    Validation accuracy: 90.0%
    Minibatch loss at step 147000: 0.830479
    Minibatch accuracy: 86.7%
    Validation accuracy: 90.1%
    Minibatch loss at step 147500: 0.634764
    Minibatch accuracy: 91.4%
    Validation accuracy: 90.0%
    Minibatch loss at step 148000: 0.666868
    Minibatch accuracy: 91.4%
    Validation accuracy: 90.0%
    Minibatch loss at step 148500: 0.684201
    Minibatch accuracy: 89.8%
    Validation accuracy: 90.0%
    Minibatch loss at step 149000: 0.795028
    Minibatch accuracy: 88.3%
    Validation accuracy: 90.0%
    Minibatch loss at step 149500: 0.726545
    Minibatch accuracy: 88.3%
    Validation accuracy: 90.0%
    Minibatch loss at step 150000: 0.650992
    Minibatch accuracy: 91.4%
    Validation accuracy: 90.0%
    Test accuracy: 95.8%
    

Minibatch loss at step 150000: 0.650992
Minibatch accuracy: 91.4%
Validation accuracy: 90.0%
Test accuracy: 95.8%
