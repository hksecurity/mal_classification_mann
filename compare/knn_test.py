import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import random

def loadTrainData(fname):
    with np.load(fname) as data:
        train = data['train']
        train_labels = data['train_labels']

    return train, train_labels

def random_class(train, train_labels, train_size = 0.3):

    key_length = len(train)
    train_size = key_length * (1 - train_size)
    # print(train_size)

    class_key_list = list()
    for i in range(int(key_length/20)):
        class_key_list.append(i)

    random.shuffle(class_key_list)
    # print(class_key_list)

    new_train = list()
    new_train_labels = list()
    new_test = list()
    new_test_labels = list()

    for key in class_key_list:
        for i, train_key in enumerate(train_labels):
            if train_key == str(key):
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

    return new_train, new_test, new_train_labels, new_test_labels

def training(npz_file):
    train, train_labels = loadTrainData(npz_file)

    (trainData, testData, trainLabels, testLabels) = random_class(train, train_labels)

    # (trainData, testData, trainLabels, testLabels) = train_test_split(train, train_labels, test_size=0.3, random_state=42)
    # (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1, random_state=84)
    # #
    print("training data points: {}".format(len(trainLabels)))
    # print("validation data points: {}".format(len(valLabels)))
    print("testing data points: {}".format(len(testLabels)))

    # print(np.shape(trainData))
    # print(np.shape(testData))
    # print(np.shape(trainLabels))
    # print(np.shape(testLabels))

    # print(trainData[:2])

    kVals = range(1, 30, 2)
    accuracies = []

    for k in range(1, 30, 2):
        # train the k-Nearest Neighbor classifier with the current value of `k`
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(trainData, trainLabels)

        # evaluate the model and update the accuracies list
        # score = model.score(valData, valLabels)
        score = model.score(testData, testLabels)
        print("k=%d, accuracy=%.2f%%" % (k, score * 100))
        accuracies.append(score)

    # find the value of k that has the largest accuracy
    i = np.argmax(accuracies)
    print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i], accuracies[i] * 100))

    model = KNeighborsClassifier(n_neighbors=kVals[i])
    model.fit(trainData, trainLabels)
    predictions = model.predict(testData)
    # print(testLabels, predictions)
    #
    print("EVALUATION ON TESTING DATA")
    print(classification_report(testLabels, predictions))

#
#
# # npz_file = 'data/mnist.npz'
# # npz_file = 'data/malimg.npz'
npz_file = '../data/mal60.npz'

training(npz_file)
#
# if b % 100 == 0:
#     x_inst, x_label, y = data_loader.fetch_batch(iv.n_classes, iv.batch_size, iv.seq_length, type='test')
#     feed_dict = {mann.x_inst: x_inst, mann.x_label: x_label, mann.y: y}
#     output, learning_loss = sess.run([mann.o, mann.learning_loss], feed_dict=feed_dict)
#     merged_summary = sess.run(mann.learning_loss_summary, feed_dict=feed_dict)
#     train_writer.add_summary(merged_summary, b)
#     accuracy = test_f(iv, y, output)
#     curve = test_f2(iv, y, output)
#
#     for accu in accuracy:
#         print('%.4f' % accu, end='\t')
#     print('%d\t%.4f' % (b, learning_loss))

