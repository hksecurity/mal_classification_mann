import numpy as np
import tensorflow as tf
from mann import cell

class memory_augmented_neural_networks():
    def __init__(self, values):
        values.output_dim = values.n_classes

        self.x_inst = tf.placeholder(dtype=tf.float32, shape=[values.batch_size,
                                                              values.seq_length,
                                                              values.insts_size])
        self.x_label = tf.placeholder(dtype=tf.float32, shape=[values.batch_size,
                                                               values.seq_length,
                                                               values.output_dim])
        self.y = tf.placeholder(dtype=tf.float32, shape=[values.batch_size,
                                                         values.seq_length,
                                                         values.output_dim])

        mann_cell = cell.mann_cell(values.rnn_size,
                                   values.memory_size,
                                   values.memory_vector_dim,
                                   head_num=values.read_head_num)
        state = mann_cell.zero_state(values.batch_size, tf.float32)
        self.state_list = [state]
        self.o = []
        for seq in range(values.seq_length):
            output, state = mann_cell(tf.concat([self.x_inst[:, seq, :], self.x_label[:, seq, :]], axis=1), state)
            with tf.variable_scope("o2o", reuse=(seq > 0)):
                o2o_w = tf.get_variable('o2o_w', [output.get_shape()[1], values.output_dim],
                                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                # initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
                o2o_b = tf.get_variable('o2o_b', [values.output_dim],
                                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                # initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
                output = tf.nn.xw_plus_b(output, o2o_w, o2o_b)

                output = tf.nn.softmax(output, dim=1)
                self.o.append(output)
                self.state_list.append(state)

        self.o = tf.stack(self.o, axis=1)
        self.state_list.append(state)

        eps = 1e-8
        self.learning_loss = -tf.reduce_mean(  # cross entropy function
            tf.reduce_sum(self.y * tf.log(self.o + eps), axis=[1, 2])
        )

        # self.learning_loss = -tf.reduce_mean(self.y * tf.log(tf.clip_by_value(self.o, eps, 1.0)) + (1-self.y) * tf.log(tf.clip_by_value(self.o, eps, 1.0)))

        # self.learning_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.o))
        # updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_var, labels=target_ph_oh),
        #                       name="cost")

        self.o = tf.reshape(self.o, shape=[values.batch_size, values.seq_length, -1])
        self.learning_loss_summary = tf.summary.scalar('learning_loss', self.learning_loss)

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=values.learning_rate)
            self.train_op = self.optimizer.minimize(self.learning_loss)