import tensorflow as tf
from  tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

# 製造容器
#Weights_1 = tf.get_variable("W1", shape = [784, 50])
#biases_1 = tf.get_variable("b1", shape = [1, 50])
#Weights_2 = tf.get_variable("W2", shape = [50, 40])
#biases_2 = tf.get_variable("b2", shape = [1, 40])
#Weights_3 = tf.get_variable("W3", shape = [40, 10])
#biases_3 = tf.get_variable("b3", shape = [1, 10])
# 開始跑
#sess = tf.Session()
#saver = tf.train.Saver()
#init = tf.global_variables_initializer()
#sess.run(init)
#saver.restore(sess, '/Github_rep/python_notebook/Tensorflow_notebook/model/DnnModel_demo')
#print('Model restored.')
#print('==================')

# Buliding Accuracy function
def compute_accuracy(v_xs, v_ys):
    global result
    y_pre = sess.run(result, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result22 = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob:1})
    return result22

# PlaceHolder
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

# Tensorflow Nerual Network Step

# Step 1 activation function = softmax

## Hidden layer 1

#Weights_1 = tf.Variable(tf.random_normal([784, 50]), name = "W1")
Weights_1 = tf.get_variable("W1", shape = [784, 50])
#biases_1 = tf.Variable(tf.constant(0.1, shape = [1, 50]), name ="b1")
biases_1 = tf.get_variable("b1", shape = [1, 50])
Wx_plus_b_1 = tf.matmul(xs, Weights_1) + biases_1
# dropout
Wx_plus_b_1 = tf.nn.dropout(Wx_plus_b_1, keep_prob)
outputs_1 = tf.nn.sigmoid(Wx_plus_b_1)

## Hidden layer 2

#Weights_2 = tf.Variable(tf.random_normal([50, 40]), name = "W2")
Weights_2 = tf.get_variable("W2", shape = [50, 40])
#biases_2 = tf.Variable(tf.constant(0.1, shape = [1, 40]), name ="b2")
biases_2 = tf.get_variable("b2", shape = [1, 40])
Wx_plus_b_2 = tf.matmul(outputs_1, Weights_2) + biases_2
# dropout
Wx_plus_b_2 = tf.nn.dropout(Wx_plus_b_2, keep_prob)
outputs_2 = tf.nn.sigmoid(Wx_plus_b_2)

## Add output layer

#Weights_3 = tf.Variable(tf.random_normal([40, 10]), name = "W3")
Weights_3 = tf.get_variable("W3", shape = [40, 10])
#biases_3 = tf.Variable(tf.constant(0.1, shape = [1, 10]), name = "b3")
biases_3 = tf.get_variable("b3", shape = [1, 10])
Wx_plus_b_3 = tf.matmul(outputs_2, Weights_3) + biases_3
# dropout
Wx_plus_b_3 = tf.nn.dropout(Wx_plus_b_3, keep_prob)
result = tf.nn.softmax(Wx_plus_b_3)

# Step 2 loss error method(loss function) = cross entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(result),
                                              reduction_indices=[1]))   
# Step 3 Gradient Descent
train_step = tf.train.GradientDescentOptimizer(0.4).minimize(cross_entropy)

# Step 4 Set session and initializer

sess = tf.Session()
#saver = tf.train.import_meta_graph('/Github_rep/python_notebook/Tensorflow_notebook/model/DnnModel_demo.ckpt.meta')
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess, '/Github_rep/python_notebook/Tensorflow_notebook/model/DnnModel_demo')
print('Model restored.')
#print('==================')
print("==============TEST==============")
print(compute_accuracy(mnist.test.images, mnist.test.labels))
print("==============END==============")