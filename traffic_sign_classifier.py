# Load pickled data
import pickle
import tensorflow as tf
import os
import numpy as np
import math
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from PIL import Image

MODEL_NAME='traffic-sign-classifier'
MODEL_PATH='./models'
EPOCHS = 30
BATCH_SIZE = 128
MAX_LEARNING_RATE=0.001
MIN_LEARNING_RATE=0.001 #learning rate decay hurt accuracy (thinking not enough epochs)
DECAY_SPEED=2000.0
PKEEP_TRAIN=0.75

# training data saved to this directory, but checked in as get_data.sh script (files too large for github)

training_file = 'data/train.p'
testing_file = 'data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
        
X, y = train['features'], train['labels']
#split the training data into 80% train and 20% validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# Number of training examples
n_train = len(X_train)

# Number of validation examples
n_validation = len(X_valid)

# Number of testing examples.
n_test = len(X_test)

# What's the shape of an traffic sign image?  We will later change this to grayscale
image_shape = X_train[0].shape

# How many unique classes/labels there are in the dataset.
n_classes = y_train.max()+1

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
def rgb2gray(rgb):
    if len(rgb.shape) == 4:
        G = np.zeros([len(rgb), rgb.shape[1], rgb.shape[2], 1])
        for i, img in enumerate(rgb):
            r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
            gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
            G[i] = gray.reshape(rgb.shape[1], rgb.shape[2],1)
        return G
    else:
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
        return gray
    
def normalize(img):
    img = rgb2gray(img)
    img = (img-128)/128
    return img

#turn RGB to grayscale images (this improved accuracy by 8%)
X_train = normalize(X_train)
X_valid = normalize(X_valid)
X_test = normalize(X_test)

image_shape = X_train[0].shape
print("New image data shape =", image_shape)

X_train, y_train = shuffle(X_train, y_train)

### Define your architecture here.
### Feel free to use as many code cells as needed.
from tensorflow.contrib.layers import flatten

def LeNet(x, pkeep):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, X_train[0].shape[2], 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    
    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)
    
    #dropout
    conv1 = tf.nn.dropout(conv1, pkeep)
    
    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)
    
    #dropout
    conv2 = tf.nn.dropout(conv2, pkeep)
    
    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)
    
    #dropout
    fc1 = tf.nn.dropout(fc1, pkeep)
    
    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)
    
    #dropout
    fc2 = tf.nn.dropout(fc2, pkeep)
    
    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits


pkeep = tf.placeholder(tf.float32, name='dropout')
x = tf.placeholder(tf.float32, (None, 32, 32, X_train[0].shape[2]))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

logits = LeNet(x, pkeep)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
learning_rate = tf.placeholder(tf.float32, name='learning-rate')
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_operation = optimizer.minimize(loss_operation)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, pkeep:1.})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


model_location='{}/{}'.format(MODEL_PATH, MODEL_NAME)
print('Model Location: ', model_location)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    batch_count = num_examples/BATCH_SIZE
    for epoch in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        offset = batch_count * epoch
        for i in range(0, num_examples, BATCH_SIZE):
            end = i + BATCH_SIZE
            iter = offset + i
            batch_x, batch_y = X_train[i:end], y_train[i:end]
            rate = MIN_LEARNING_RATE + (MAX_LEARNING_RATE - MIN_LEARNING_RATE) * math.exp(-iter/DECAY_SPEED)
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, learning_rate: rate, pkeep:PKEEP_TRAIN})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(epoch+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, model_location)
    print("Model saved")

    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loader = tf.train.import_meta_graph('{}.meta'.format(model_location))
        loader.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))
        sess = tf.get_default_session()
        
        test_accuracy = evaluate(X_test, y_test)
        print("Test Accuracy = {:.3f}".format(test_accuracy))

size = 32, 32
softmax_prob = None

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loader = tf.train.import_meta_graph('{}/{}.meta'.format(MODEL_PATH, MODEL_NAME))
    loader.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))
    sess = tf.get_default_session()
    
    i = 0
    for file in os.listdir('new_image_data'):
        if '.jpg' in file:
            image = Image.open('new_image_data/' + file)
            image.thumbnail((32,32), Image.ANTIALIAS)
            image = normalize(np.array([np.array(image)]))
            softmax = tf.nn.softmax(logits)
            top5 = tf.nn.top_k(softmax, k=5, sorted=True)
            softmax_prob, top5_val = sess.run([softmax, top5], feed_dict={x:image, pkeep:1.})
            print('Image [{}.jpg]: {}'.format(i, top5_val))
