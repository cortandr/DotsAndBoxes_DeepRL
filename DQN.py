import numpy as np
import tensorflow as tf
import random
import copy


class DQN:
    """DQN agent class"""

    def __init__(self, input_shape, training, alpha=1e-4, gamma=0.7):

        self.input_shape = input_shape

        # Variables needed for Q function interaction
        self.training = training
        self.input_layer = None
        self.sess = None
        self.Q_values = None
        self.target_Q = None
        self.loss = None
        self.saver = None
        self.gamma = gamma  # discount rate
        self.epsilon = 0.7  # exploration rate
        self.alpha = alpha
        self.optimizer = None
        self.train_op = None
        self.next_state = None
        self.sess = None

        # Build network
        self.build_network()

    def predict(self,feature_vector):

        """
        Prediction method used to predict best action given the current state
        :param feature_vector: current state
        :return:
        """

        # Generate all allowed moves from state
        actions = self.generateMoves(feature_vector)
        pairs = []

        # Compute all the Q values for those actions
        for a in actions:

            f = copy.deepcopy(feature_vector)
            next_f = copy.deepcopy(feature_vector)

            # Unpack current action
            i, j, d = a

            # Map action to int in array
            if d == 'h':
                next_f[i][j][0] = 1
            else:
                next_f[i][j][1] = 1

            # Prepare input tensor and run model
            f = np.append(f, next_f, axis=2)
            f = np.reshape(f, (1, self.input_shape+1, self.input_shape+1, 4))
            q = self.sess.run(self.Q_values, feed_dict={self.input_layer: f})
            pairs.append((a, q))

        # Roll the dice:
        # Take random move or choose best Q-value and associated action
        if self.training == True and random.uniform(0, 1) > self.epsilon:
            i = random.randint(0, len(pairs)-1)
            return pairs[i][0]
        else:
            best_q = 0
            best_action_q_pair = ()
            for pair in pairs:
                if best_q == 0:
                    best_q = pair[1]
                    best_action_q_pair = pair[0]
                elif pair[1] > best_q:
                    best_action_q_pair = pair[0]
                    best_q = pair[1]
            return best_action_q_pair

    def build_network(self):

        """
        Builds network and initializes all tensorflow related variables
        :return:
        """

        g = tf.Graph()
        with g.as_default():

            # Input and Target Layers
            self.input_layer = tf.placeholder(tf.float32, [None, None, None, 4])
            self.target_Q = tf.placeholder(tf.float32, (1,1))

            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(
                inputs=self.input_layer,
                filters=32,
                kernel_size=[3, 3],
                strides = 1,
                padding="same",
                activation=tf.nn.relu,
                name = "conv1")

            # Convolutional Layer #2
            conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=64,
                kernel_size=[3, 3],
                strides = 1,
                padding="same",
                activation=tf.nn.relu,
                name = "conv2")

            # Convolutional Layer #3
            conv3 = tf.layers.conv2d(
                inputs=conv2,
                filters=32,
                kernel_size=[3, 3],
                strides=1,
                padding="same",
                activation=tf.nn.relu,
                name="conv3")

            # Averaging pooling Layer
            avPool_out = tf.reduce_mean(conv3, axis=(1,2), keepdims=True)
            avPool_output = tf.layers.flatten(avPool_out)
            normalized_avPool_output = tf.layers.batch_normalization(avPool_output,
                                                                     axis=1)

            # FCN

            # First layer
            fc_1 = tf.layers.dense(inputs=normalized_avPool_output,
                                   units=33,
                                   activation=tf.nn.elu, name = "fc_nn1")

            normalized_fc_1 = tf.layers.batch_normalization(fc_1, axis=1)

            dropout_fc_1 = tf.layers.dropout(normalized_fc_1, rate=0.5)

            # Second layer
            fc_2 = tf.layers.dense(inputs=dropout_fc_1,
                                   units=30,
                                   activation=tf.nn.elu, name = "fc_nn2")

            normalized_fc_2 = tf.layers.batch_normalization(fc_2, axis=1)

            dropout_fc_2 = tf.layers.dropout(normalized_fc_2, rate=0.5)

            self.Q_values = tf.layers.dense(inputs=dropout_fc_2,
                                            units=1,
                                            activation=tf.nn.tanh,
                                            name="q_values")

            # Calculate Loss
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q_values))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha)
            self.train_op = self.optimizer.minimize(loss=self.loss)

            # Initialize all variables
            init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            self.sess = tf.Session()
            self.sess.run(init)

    def generateMoves(self, feature_vector):

        """
        Method used to generate all allowed moves given the current state
        :param feature_vector: current state represented as a matrix
        :return: list of tuple with coordinates of allowed moves
        """
        moves = []
        for i in range(len(feature_vector)):
            for j in range(len(feature_vector)):
                if i == len(feature_vector)-1 and j == len(feature_vector)-1:
                    return moves
                if feature_vector[i][j][1] == 0 and i < len(feature_vector)-1:
                    moves.append((i, j, 'v'))
                if feature_vector[i][j][0] == 0 and j < len(feature_vector)-1:
                    moves.append((i, j, 'h'))

    def save_model(self, path):

        """
        Save model
        :param path: path to weights file
        :return: path to saved model
        """
        p = self.saver.save(self.sess, (path + "/model"))
        return p

    def load_model(self, path):

        """
        Load model
        :param path: path to restore the model from
        :return:
        """
        self.saver.restore(self.sess, path)
