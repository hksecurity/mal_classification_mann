import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# npz_file = 'data/mnist.npz'
# nb_classes = 10
npz_file = 'data/malimg.npz'
nb_classes = 25
# npz_file = 'data/mal60.npz'
# nb_classes = 60

def one_hot_encode(data):

    targets = data.reshape(-1)
    targets = np.array(targets, dtype='i')
    one_hot_targets = np.eye(nb_classes)[targets]

    return one_hot_targets

def loadTrainData(fname):
    with np.load(fname) as data:
        train = data['train']
        train_labels = data['train_labels']

        train_labels = train_labels.flatten()
        train_labels = one_hot_encode(train_labels)

    return train, train_labels

train, train_labels = loadTrainData(npz_file)

(trainData, testData, trainLabels, testLabels) = train_test_split(train, train_labels, test_size=0.25, random_state=42)

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


#  STRART!!!

print("training data points: {}".format(len(trainLabels)))
print("testing data points: {}".format(len(testLabels)))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 20
num_samples = len(trainData)

total_batch = int(num_samples / batch_size)
batch_pointer = batch_size

for epoch in range(30):
    total_cost = 0

    for i in range(total_batch):
        batch_xs = trainData[:batch_pointer]
        batch_ys = trainLabels[:batch_pointer]

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

        batch_pointer = batch_pointer + batch_size

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.10f}'.format(total_cost / total_batch))

print('최적화 완료!')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy,
                        feed_dict={X: testData,
                                   Y: testLabels}))