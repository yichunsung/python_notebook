import tensorflow as tf
import pandas as pd
import numpy as np
# Data loading
data  = pd.read_csv("train70.csv", sep = ",", dtype=None)
testData = pd.read_csv("train30.csv", sep = ",", dtype=None)
#print(data.head())
label = pd.read_csv("train70Label.csv")
testlabel= pd.read_csv("train30Label.csv")
#data = data.astype(np.float)

#print('data({0[0]},{0[1]})'.format(data.shape))
#print(data.dtype)
#print(data.head())

# Function definition
## Filter function
def filter_weight(filter_shape):
    filter_ForConvLayer = tf.truncated_normal(filter_shape, stddev = 0.1)
    return tf.Variable(filter_ForConvLayer)

## Bias function
def bias(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

## Convoluation layer function
def add_ConvolutionLayer(inputData, filter_weight, activation_function = None):
    conv2dLayer = tf.nn.conv2d(input = inputData, 
                               # filter should be [filter_hight, filter_width, in_channels, out_channels]
                               filter = filter_weight,
                               strides = [1, 2, 2,1], # strides = [1, x_strides, y_strides, 1]
                               padding = 'SAME'
                               )
    if activation_function is None:
        output_result = conv2dLayer
    else:
        output_result = activation_function(conv2dLayer)
    return output_result

## Max pooling layer function
def add_MaxPoolingLayer(inputData):
    MaxPooling = tf.nn.max_pool(inputData, # input data should be [batch, height, width, channels] 
                                ksize = [1, 2, 2, 1], # 2*2 pixels for max pooling
                                strides = [1, 2, 2, 1], # strides = [1, x_strides, y_strides, 1]
                                padding = 'SAME'
                                )
    
    return MaxPooling

## Flatten layer function
def add_FlattenLayer(inputData, numbersOfFactors):
    flattenLayer = tf.reshape(inputData, [-1, numbersOfFactors])
    return flattenLayer

## Fully connected layer function
def add_FullyConnectedLayer(inputData, Weight_FCLayer, bias_FCLayer, activation_function = None):
    Wx_plus_b = tf.matmul(inputData, Weight_FCLayer)+bias_FCLayer
    if activation_function is None:
        fullyConnectedLayer = Wx_plus_b
    else:
        fullyConnectedLayer = activation_function(Wx_plus_b)
    return fullyConnectedLayer

def compute_accuracy(v_xs, v_ys):
    global output_layer
    y_pre = sess.run(output_layer, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


# Model building

# Step 0 -- Placeholder definition    
xs = tf.placeholder(tf.float32, [None, 784]) # Input data = 28*28(= 784 factors) pixels image
ys = tf.placeholder(tf.float32, [None, 10])  # Output result = 0~9 labels 
keep_prob = tf.placeholder(tf.float32) # Dropout probability
x_image = tf.reshape(xs, [-1, 28, 28, 1])

## Step 1 -- Convolution layer 1
### Filter = 5*5 , chanel = 1 (black and white), 32 filters
filter_convLayer_1 = filter_weight([5, 5, 1, 32]) 
layer_1_convolution = add_ConvolutionLayer(
    inputData = x_image, 
    filter_weight = filter_convLayer_1,
    activation_function = tf.nn.relu
    )
## Stap 2 -- Max pooling layer 1
layer_1_MaxPoooling = add_MaxPoolingLayer(layer_1_convolution)

## Step 3 -- Convolution 1ayer 2
### Filter = 4*4, chanel = 32, 64 filters
filter_convLayer_2 = filter_weight([4, 4, 32, 64])
layer_2_convolution = add_ConvolutionLayer(
    inputData = layer_1_MaxPoooling, 
    filter_weight = filter_convLayer_2,
    activation_function = tf.nn.relu
    )

## Step 4 -- Max pooling layer 2
layer_2_MaxPoooling = add_MaxPoolingLayer(layer_2_convolution)

## Step 5 -- Flatten layer
flatten = add_FlattenLayer(layer_2_MaxPoooling, 2*2*64)

## Step 6 -- Fully connected Layer 1
fc_layer_1 = add_FullyConnectedLayer(
    inputData = flatten,
    Weight_FCLayer =  tf.Variable(tf.random_normal([2*2*64, 1024], stddev = 0.1)),
    bias_FCLayer = bias([1024]),
    activation_function = tf.nn.relu
    )
## Step 7 -- Output layer/Fully connected layer 2
output_layer = add_FullyConnectedLayer(
    inputData = fc_layer_1, 
    Weight_FCLayer =  tf.Variable(tf.random_normal([1024, 10], stddev = 0.1)),
    bias_FCLayer = bias([10]),
    activation_function = tf.nn.softmax
    )

# Model Training 
## loss function (cross entropy)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(output_layer),
                                              reduction_indices=[1]))  
## train step
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# Set Session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Run
for i in range(1000):
    #batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: testData, ys: testlabel})
    if i % 50 == 0:
    	print(sess.run(cross_entropy, feed_dict={xs: testData, ys: testlabel}))
    #    print(compute_accuracy(testData, testlabel))

#print(filter_convLayer_1)