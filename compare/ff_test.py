import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import random

# npz_file = 'data/mnist.npz'
# nb_classes = 10
npz_file = '../data/malimg.npz'
nb_classes = 25
# npz_file = '../data/mal60.npz'
# nb_classes = 60
batch_size = 16
sample_count = 50
total_epoch = 10000

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

    #print(np.shape(new_train))

    return new_train, new_test, new_train_labels, new_test_labels

def loadTrainData(fname):
    with np.load(fname) as data:
        train = data['train']
        train_labels = data['train_labels']

        train_labels = train_labels.flatten()
        train_labels = one_hot_encode(train_labels)

    return train, train_labels

train, train_labels = loadTrainData(npz_file)

(trainData, testData, trainLabels, testLabels) = random_class(train, train_labels)
# (trainData, testData, trainLabels, testLabels) = train_test_split(train, train_labels, test_size=0.5, random_state=42)

X = tf.placeholder(tf.float32, [None, 400])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W1 = tf.Variable(tf.random_normal([400, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_normal([256, nb_classes], stddev=0.01))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


def test_f(y, output):
    correct = [0] * sample_count
    total = [0] * sample_count
    y_decode = y
    output_decode = output

    # print(np.shape(y)[0])

    for i in range(np.shape(y)[0]):
        y_i = y_decode[i]
        output_i = output_decode[i]
        class_count = {}
        for j in range(sample_count):
            if y_i[j] not in class_count:
                class_count[y_i[j]] = 0
            class_count[y_i[j]] += 1
            total[class_count[y_i[j]]] += 1
            if y_i[j] == output_i[j]:
                correct[class_count[y_i[j]]] += 1

    return [float(correct[i]) / total[i] if total[i] > 0. else 0. for i in range(1, 11)]

#  STRART!!!

print("training data points: {}".format(len(trainLabels)))
print("testing data points: {}".format(len(testLabels)))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

num_samples = len(trainData)

total_batch = int(num_samples / batch_size)
batch_pointer = batch_size
before_batch_pointer = 0
print(total_batch)

for epoch in range(total_epoch):
    total_cost = 0

    if epoch % 100 == 0:

        output = tf.argmax(model, 1)
        y_output = tf.argmax(Y, 1)

        # accuracy = test_f(y_output, output)
        output = sess.run(output, feed_dict={X: testData, Y: testLabels})
        y_output = sess.run(y_output, feed_dict={X: testData, Y: testLabels})

        class_key_list = list()
        for i in range(len(output)):
            class_key_list.append(i)

        new_output = list()
        new_y_output = list()

        for b in range(batch_size):
            random.shuffle(class_key_list)
            temp_list = list()
            temp_y_list = list()

            for key in class_key_list:
                if len(temp_list) < sample_count:
                    temp_list.append(output[key])
                    temp_y_list.append(y_output[key])

            new_output.append(temp_list)
            new_y_output.append(temp_y_list)

        # print(np.shape(new_output))
        accuracy = test_f(new_output, new_y_output)

        for accu in accuracy:
            print('%.4f' % accu, end='\t')
        print('%d' % epoch)

    for i in range(total_batch):

        batch_xs = trainData[before_batch_pointer:batch_pointer]
        batch_ys = trainLabels[before_batch_pointer:batch_pointer]

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

        batch_pointer = batch_pointer + batch_size

    before_batch_pointer = 0
    batch_pointer = batch_size

    if epoch % 100 == 0:
        print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.10f}'.format(total_cost / total_batch))


is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
output = tf.argmax(model, 1)
y_output = tf.argmax(Y, 1)
# print('model', model)
# print('sess.run', sess.run(model, feed_dict={X: testData, Y: testLabels}))
# print('is_correct', is_correct)
 # print('output', sess.run(output, feed_dict={X: testData, Y: testLabels}))

output = sess.run(output, feed_dict={X: testData, Y: testLabels})
y_output = sess.run(y_output, feed_dict={X: testData, Y: testLabels})
# print('output', np.shape(output))
# print('y_output', np.shape(y_output))

# print('y_output', sess.run(y_output, feed_dict={X: testData,
#                                    Y: testLabels}))
# print('sess.run2', sess.run(is_correct, feed_dict={X: testData,
#                                    Y: testLabels}))
#
#
# output2 = tf.Print(output, [output], message="output: ")
# print(output2)
# output = tf.argmax(model, 1)

# y_output = tf.argmax(Y, 1)
# y_output = tf.Print(y_output, [y_output], message="y_output: ")
# sess.run(output, feed_dict={X: testData, Y: testLabels})
# sess.run(y_output, feed_dict={X: testData, Y: testLabels})
# is_correct = tf.Print(is_correct, len([is_correct]), message="test ")
# sess.run(is_correct, feed_dict={X: testData, Y: testLabels})
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy,
                        feed_dict={X: testData,
                                   Y: testLabels}))