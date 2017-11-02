import tensorflow as tf
from  tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

# Buliding add_layer function
def add_Hidden_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    #biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    biases = tf.Variable(tf.constant(0.1, shape = [1, out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # dropout
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
    
def add_output_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    #biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    biases = tf.Variable(tf.constant(0.1, shape = [1, out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
# Buliding Accuracy function
def compute_accuracy(v_xs, v_ys):
    global output_Layer
    y_pre = sess.run(output_Layer, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob:1})
    return result

# PlaceHolder
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

# Tensorflow Nerual Network Step

# add_layer(inputs, in_size, out_size, activation_function = None)
# Step 1 activation function = softmax

## Hidden layer 1
hidden_Layer1 = add_Hidden_layer(xs, 784, 50, activation_function = tf.nn.sigmoid)
## Hidden layer 2
hidden_Layer2 = add_Hidden_layer(hidden_Layer1, 50, 40, activation_function = tf.nn.sigmoid)
## Add output layer
output_Layer = add_output_layer(hidden_Layer2, 40, 10, activation_function = tf.nn.softmax)

# Step 2 loss error method(loss function) = cross entropy
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(output_Layer),
#                                              reduction_indices=[1]))   
# Step 3 Gradient Descent
#train_step = tf.train.GradientDescentOptimizer(0.4).minimize(cross_entropy)

# Step 4 Set session and initializer
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.import_meta_graph('/Github_rep/python_notebook/Tensorflow_notebook/model/DnnModel_test.ckpt.meta')
#saver = tf.train.Saver()
saver.restore(sess, '/Github_rep/python_notebook/Tensorflow_notebook/model/DnnModel_test.ckpt')
print('==================')
print(compute_accuracy(mnist.test.images, mnist.test.labels))