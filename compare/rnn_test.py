import tensorflow as tf
import numpy as np
import random

# npz_file = '../data/mnist.npz'
# nb_classes = 10
# npz_file = '../data/malimg.npz'
# nb_classes = 25
npz_file = '../data/mal60.npz'
nb_classes = 60
sample_count = 50

learning_rate = 0.001
total_epoch = 100000
batch_size = 128

n_input = 20
n_step = 20
n_hidden = 128


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

    # print(np.shape(new_train))

    return new_train, new_test, new_train_labels, new_test_labels


def loadTrainData(fname):
    with np.load(fname) as data:
        train = data['train']
        train_labels = data['train_labels']

        train_labels = train_labels.flatten()
        train_labels = one_hot_encode(train_labels)

    return train, train_labels


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


X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([n_hidden, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

# cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


data, data_labels = loadTrainData(npz_file)
(trainData, testData, trainLabels, testLabels) = random_class(data, data_labels)

# print(np.shape(trainData))
# print(np.shape(trainLabels))
# print(np.shape(testData))
# print(np.shape(testLabels))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
num_samples = len(trainData)
batch_pointer = batch_size
before_batch_pointer = 0

total_batch = int(num_samples / batch_size)
# print('total_batch', total_batch)

for epoch in range(total_epoch):
    total_cost = 0

    if epoch % 10 == 0:

        output = tf.argmax(model, 1)
        y_output = tf.argmax(Y, 1)
        test_batch_size = len(testLabels)
        test_xs = testData.reshape(test_batch_size, n_step, n_input)
        test_ys = testLabels
        output = sess.run(output, feed_dict={X: test_xs, Y: test_ys})
        y_output = sess.run(y_output, feed_dict={X: test_xs, Y: test_ys})
        # print(np.shape(output))

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
        # print('batch_xs', i, before_batch_pointer, batch_pointer, np.shape(batch_xs))
        # print('batch_xs', np.shape(batch_xs))
        # print('batch_ys', np.shape(batch_ys))
        batch_xs = batch_xs.reshape((batch_size, n_step, n_input))
        # print('after batch_xs', np.shape(batch_xs))

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val
        before_batch_pointer = batch_pointer
        batch_pointer = batch_pointer + batch_size

    before_batch_pointer = 0
    batch_pointer = batch_size

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.4f}'.format(total_cost / total_batch))

print('최적화 완료!')


is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

test_batch_size = len(testLabels)
# print(test_batch_size)
test_xs = testData.reshape(test_batch_size, n_step, n_input)
test_ys = testLabels

print('test_batch_size', test_batch_size)
print('test_xs', np.shape(test_xs))
print('test_ys', np.shape(test_ys))

print('정확도:', sess.run(accuracy, feed_dict={X: test_xs, Y: test_ys}))
