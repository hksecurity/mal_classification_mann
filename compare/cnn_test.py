import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import random

# npz_file = 'data/mnist.npz'
# nb_classes = 10
# npz_file = 'data/malimg.npz'
# nb_classes = 25
npz_file = '../data/mal60.npz'
nb_classes = 60

def one_hot_encode(data):

    targets = data.reshape(-1)
    targets = np.array(targets, dtype='i')
    one_hot_targets = np.eye(nb_classes)[targets]

    return one_hot_targets

def one_hot_encode2(data):

    targets = np.array(data, dtype='i')
    one_hot_targets = np.eye(nb_classes)[targets]

    return one_hot_targets


def random_class(train, train_labels, train_size = 0.3):

    key_length = len(train)
    train_size = key_length * (1 - train_size)
    # print(train_size)

    class_key_list = list()
    for i in range(int(key_length/20)):
        class_key_list.append(i)

    class_key_list = one_hot_encode2(class_key_list)

    random.shuffle(class_key_list)

    new_train = list()
    new_train_labels = list()
    new_test = list()
    new_test_labels = list()

    for key in class_key_list:
        for i, train_key in enumerate(train_labels):
            # print(key, train_key)
            if (train_key == key).all():
                if len(new_train) < int(train_size):
                    new_train.append(train[i])
                    new_train_labels.append(train_key)
                else:
                    new_test.append(train[i])
                    new_test_labels.append(train_key)

    # print(new_train)
    new_train = np.array(new_train)
    new_train_labels = np.array(new_train_labels)
    new_test = np.array(new_test)
    new_test_labels = np.array(new_test_labels)

    print(np.shape(new_train))

    return new_train, new_test, new_train_labels, new_test_labels


def loadTrainData(fname):
    with np.load(fname) as data:
        train = data['train']
        train_labels = data['train_labels']

        train_labels = train_labels.flatten()
        train_labels = one_hot_encode(train_labels)

    return train, train_labels

data, data_labels = loadTrainData(npz_file)

(trainData, testData, trainLabels, testLabels) = random_class(data, data_labels)
# (trainData, testData, trainLabels, testLabels) = train_test_split(data, data_labels, test_size=0.5, random_state=42)

X = tf.placeholder(tf.float32, [None, 20, 20, 1])
Y = tf.placeholder(tf.float32, [None, nb_classes])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W3 = tf.Variable(tf.random_normal([5 * 5 * 64, 256], stddev=0.01))
L3 = tf.reshape(L2, [-1, 5 * 5 * 64])
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob)

W4 = tf.Variable(tf.random_normal([256, nb_classes], stddev=0.01))
model = tf.matmul(L3, W4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)


print("training data points: {}".format(len(trainLabels)))
# print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 20
num_samples = len(trainData)

total_batch = int(num_samples / batch_size)
batch_pointer = batch_size

for epoch in range(100):
    total_cost = 0

    for i in range(total_batch):
        batch_xs = trainData[:batch_pointer]
        batch_ys = trainLabels[:batch_pointer]

        batch_xs = batch_xs.reshape(-1, 20, 20, 1)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
        total_cost += cost_val
        batch_pointer = batch_pointer + batch_size

    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.10f}'.format(total_cost / total_batch))

print('Complete')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy, feed_dict={X: testData.reshape(-1, 20, 20, 1), Y: testLabels, keep_prob: 1}))