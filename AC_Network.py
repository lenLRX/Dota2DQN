import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


class AC_Network():

    def __init__(self, s_size, a_size, hidden_size, trainer=tf.train.AdamOptimizer(learning_rate=1e-4)):
        self.s_size = s_size
        self.a_size = a_size
        self.hidden_size = hidden_size
        self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32, name = "state_input")
        self.W1 = self.weight_variable([self.s_size, self.hidden_size])
        self.b1 = self.bias_variable([self.hidden_size])
        self.hidden_layer = tf.nn.relu(
            tf.matmul(self.inputs, self.W1) + self.b1)

        self.W_policy = self.weight_variable([self.hidden_size, self.a_size])
        self.b_policy = self.bias_variable([self.a_size])
        self.policy = tf.nn.softmax(
            tf.matmul(self.hidden_layer, self.W_policy) + self.b_policy
        )

        self.W_value = self.weight_variable([self.hidden_size, 1])
        self.b_value = self.bias_variable([1])
        self.value = tf.matmul(self.hidden_layer, self.W_value) + self.b_value
        #self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name = "action_input")
        #self.actions_onehot = tf.one_hot(
        #    self.actions, self.a_size, dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None, 3], dtype=tf.float32, name = "action_input")
        self.target_v = tf.placeholder(shape=[None, 1], dtype=tf.float32, name = "target_v")
        self.advantages = tf.placeholder(shape=[None, 1], dtype=tf.float32, name = "advantages")

        self.responsible_outputs = tf.reduce_sum(
            self.policy * self.actions, reduction_indices = 1)

        # Loss functions
        self.value_loss = 0.5 * \
            tf.reduce_sum(tf.square(self.target_v -
                                    tf.reshape(self.value, [-1])))
        self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 0.0001))
        self.policy_loss = - \
            tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
        self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

        # gradients
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.gradients = tf.gradients(self.loss, self.vars)
        self.var_norms = tf.global_norm(self.vars)
        self.grads, self.grad_norms = tf.clip_by_global_norm(
            self.gradients, 40.0)
        self.apply_grads = trainer.apply_gradients(zip(self.grads, self.vars))

    def weight_variable(self, shape):
        initial = tf.constant(0.001, shape=shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)
