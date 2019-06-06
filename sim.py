from DQN import DQN
import numpy as np
import copy
import random


class Simulation:

    """Simulation and training class"""

    def __init__(self, input_shape, replay_tbl_size=50000, update_size=100):

        self.replay_size = replay_tbl_size
        self.update_size = update_size
        self.update_frozen = 500
        self.current_replay_size = 0
        self.player1 = None
        self.player2 = None
        self.p1Score = 0
        self.prevScoreP1 = 0
        self.prevScoreP2 = 0
        self.p2Score = 0
        self.prevState = None
        self.state = self.init_state(input_shape)
        self.simNo = 1000000
        self.gameEnded = False
        self.action = None
        self.won = 0

        self.completeBoxes = self.init_complete_boxes(input_shape)
        self.completed = False
        self.path = "./saved_model4x4.FINAL"
        self.learnerWinRate = 0
        self.learnerWonGames = 0
        self.replay_table = []
        self.game_replay_table = []
        self.input_shape = input_shape

    def play(self):

        # Instantiate players
        self.player1 = DQN(self.input_shape, True)
        # self.player1.load_model(self.path + "/model")
        self.player2 = DQN(self.input_shape, False)
        # self.player2.load_model(self.path + "/model")

        i = 0
        # Run simulations
        while i < self.simNo:
            i += 1

            # Decide randomly who starts
            starting_player = random.randint(1, 2)
            current_turn = starting_player

            played = False

            # Play game
            while not self.gameEnded:

                # Adapt state for network's input
                feature_vector = self.change_state_dimension(
                    copy.deepcopy(self.state),
                    self.input_shape
                )

                if current_turn == 1:

                    # if in the previous turn player completed a box or
                    # is the agent turn again after letting the opponent play
                    if self.completed is True or played is True:

                        # Prepare data to be stored in the replay table
                        # Current state
                        s = self.change_state_dimension(
                                        copy.deepcopy(self.prevState),
                                        self.input_shape)

                        # Future state after applying action
                        next_s = self.change_state_dimension(
                                        copy.deepcopy(self.state),
                                        self.input_shape)

                        # Store in replay table
                        self.store_transition(
                                        s,
                                        copy.deepcopy(self.action),
                                        ((self.p1Score-self.p2Score)-
                                            (self.prevScoreP1-self.prevScoreP2))
                                        /(self.input_shape**2),
                                        next_s
                                        )

                        self.completed = False

                    self.prevState = copy.deepcopy(self.state)
                    self.prevScoreP1 = copy.deepcopy(self.p1Score)
                    self.prevScoreP2 = copy.deepcopy(self.p2Score)

                    # Choose best action
                    self.action = self.player1.predict(feature_vector)
                    # Apply chosen action
                    self.apply_action(self.action, 1)

                    played = True

                    # Assign turn if boxes have been completed
                    if self.completed is True:
                        current_turn = 1
                    else:
                        current_turn = 2
                else:
                    # Opponent's turn - choose action
                    a = self.player2.predict(feature_vector)

                    # Apply action
                    self.apply_action(a, 2)

                    # Assign next turn
                    if self.completed is True:
                        current_turn = 2
                    else:
                        current_turn = 1

            # Prepare data to be stored in the replay table
            # Current State
            s = self.change_state_dimension(
                                        copy.deepcopy(self.prevState),
                                        self.input_shape)
            next_s = self.change_state_dimension(
                                        copy.deepcopy(self.state),
                                        self.input_shape)
            reward = self.get_reward()
            self.store_transition(
                            s,
                            copy.deepcopy(self.action),
                            reward,
                            next_s,
                            )
            self.completed = False

            # Perform training
            if self.current_replay_size > self.update_size and i % 5 == 0:
                    self.replay()

            # Update target network with most recent weights
            if i % self.update_frozen == 0:
                p = self.player1.save_model(self.path)
                self.player2.load_model(p)
                print("Updating frozen network...")

            if i % 100 == 0:
                print(str(i) + " games have been played")

            self.reset_game(self.input_shape)

        self.player1.save_model(self.path)

    def apply_action(self, action, player):

        """
        Take step in the game by applying a chosen action
        :param action: action tuple (x, y, d)
        :param player: player id
        :return:
        """

        # Unpack action
        r, c, o = action

        # Assigned segment ob board to the player
        self.state[r][c][o] = player

        # Check for end of game
        if self.complete_box(action, player) and self.gameEnded is False:
            if self.p1Score + self.p2Score >= self.input_shape**2:
                if self.p1Score > self.p2Score:
                    self.won = 1
                    self.learnerWonGames += 1
                elif self.p1Score == self.p2Score:
                    self.won = 0
                else:
                    self.won = 2
                self.gameEnded = True

        # Update boxes counts
        self.update_boxes(action)

    # Method to store transition in the replay table
    def store_transition(self, s, a, r, next_s):

        """

        :param s: current state
        :param a: action
        :param r: reward
        :param next_s: next state
        :return:
        """

        # Build transition array
        transition = np.array([s, a, r, next_s])

        # Store based on end of game
        if self.gameEnded is True:
            self.game_replay_table.append(transition)

            # Refresh replay table if size limit has been exceeded
            if self.current_replay_size < self.replay_size:
                self.replay_table = self.replay_table + self.game_replay_table
                self.current_replay_size += len(self.game_replay_table)
                self.game_replay_table = []
            else:
                del(self.replay_table[:1999])
                self.current_replay_size -= 2000
                self.replay_table = self.replay_table + self.game_replay_table
                self.current_replay_size += len(self.game_replay_table)
                self.game_replay_table = []

        else:
            self.game_replay_table.append(transition)

    def replay(self):

        """
        Implements training epoch for neural network
        :return:
        """

        # Sample batch fro replay memory
        mini_batch = random.sample(self.replay_table, self.update_size)

        for transition in mini_batch:

            # Unpack transition
            state, action, reward, next_state = transition
            state_action = copy.deepcopy(state)

            # Unpack action coordinates
            r, c, d = action
            if d == 'h':
                state_action[r][c][0] = 1
            else:
                state_action[r][c][1] = 1

            # Prepare model input tensor
            net_input = np.append(state, state_action, axis=2)
            net_input = np.array([net_input])

            # Compute Q value of current training network
            q = self.player1.sess.run(
                            self.player1.Q_values,
                            feed_dict={self.player1.input_layer: net_input})

            # If the next state is a terminal state the target
            # is equal to the immediate reward, otherwise target update formula
            if self.end_state(next_state):

                target = reward
                target = np.reshape(target, newshape=(1, 1))

                # Train network with target_f
                l, _ = self.player1.sess.run(
                    [self.player1.loss, self.player1.train_op],
                    feed_dict={
                        self.player1.input_layer: net_input,
                        self.player1.target_Q: target,
                        self.player1.Q_values: q
                    }
                )
            else:
                target = reward + self.player1.gamma*(np.amax(
                    self.predict_q_next(next_state)))
                target = np.reshape(target, newshape=(1,1))

                # Train network with target_f
                l, _ = self.player1.sess.run(
                    [self.player1.loss, self.player1.train_op],
                    feed_dict={
                        self.player1.input_layer: net_input,
                        self.player1.target_Q: target,
                        self.player1.Q_values: q
                    }
                )

    def predict_q_next(self, n_s):

        """
        Used to predict value function for next state
        :param n_s: next state
        :return: list of Q Values
        """

        # Generate allowed moves
        possible_moves = self.generate_moves(n_s)

        # Predict q values for all possible actions in next state
        q_next = np.array([])
        for m in possible_moves:

            next_state = copy.deepcopy(n_s)
            future_state = copy.deepcopy(n_s)

            # Prepare tensor for model input
            i, j, d = m
            if d == 'h':
                future_state[i][j][0] = 1
            else:
                future_state[i][j][1] = 1

            # Prepare tensor for model input
            next_state = np.append(next_state, future_state, axis=2)
            next_state = np.array([next_state])

            # Get Q value
            q_n = self.player2.sess.run(
                            self.player2.Q_values,
                            feed_dict={self.player2.input_layer: next_state
                                       })
            q_next = np.append(q_next, q_n)

        return q_next

    def generate_moves(self,feature_vector):

        """
        Method used to generate all allowed moves
        :param feature_vector: state
        :return:
        """
        moves = []
        for i in range(self.input_shape+1):
            for j in range(self.input_shape+1):
                if i == self.input_shape and j == self.input_shape:
                    return moves
                if feature_vector[i][j][1] == 0 and i < len(feature_vector)-1:
                    moves.append((i, j, 'v'))
                if feature_vector[i][j][0] == 0 and j < len(feature_vector)-1:
                    moves.append((i, j, 'h'))

    def complete_box(self, action, player):

        """
        Method used to check for completed boxes after application of an action
        If also assigns points to the player based on completion
        :param action: chosen action
        :param player: player id
        :return: bool
        """

        # Unpack action
        r, c, o = action
        if o == 'v':
            try:
                if self.completeBoxes[self.f(r, c)] == 3 \
                                            and c != self.input_shape:
                    if player == 1:
                        self.p1Score += 1
                    else:
                        self.p2Score += 1
                    self.completed = True
                    try:
                        if self.completeBoxes[self.f(r, c) - 1] == 3 and c != 0:
                            if player == 1:
                                self.p1Score += 1
                            else:
                                self.p2Score += 1
                            self.completed = True
                            return True
                        self.completed = True
                        return True
                    except KeyError:
                        pass
                    self.completed = True
                    return True
                elif self.completeBoxes[self.f(r, c) - 1] == 3 \
                                        and c != 0 \
                                        and c != self.input_shape:
                    if player == 1:
                        self.p1Score += 1
                    else:
                        self.p2Score += 1
                    self.completed = True
                    return True
                elif self.completeBoxes[self.f(r, c) - 1] == 3 \
                                            and c == self.input_shape:
                    if player == 1:
                        self.p1Score += 1
                    else:
                        self.p2Score += 1
                    self.completed = True
                    return True
                else:
                    self.completed = False
                    return False
            except KeyError:
                try:
                    if self.completeBoxes[self.f(r, c) - 1] == 3 and c != 0:
                        if player == 1:
                            self.p1Score += 1
                        else:
                            self.p2Score += 1
                        self.completed = True
                        return True
                    else:
                        self.completed = False
                        return False
                except KeyError:
                    pass
        elif o == 'h':
            try:
                if self.completeBoxes[self.f(r, c)] == 3:
                    if player == 1:
                        self.p1Score += 1
                    else:
                        self.p2Score += 1
                    self.completed = True
                    try:
                        if self.completeBoxes[self.f(r, c) - self.input_shape] == 3:
                            if player == 1:
                                self.p1Score += 1
                            else:
                                self.p2Score += 1
                            self.completed = True
                            return True
                    except KeyError:
                        pass
                    self.completed = True
                    return True
                elif self.completeBoxes[self.f(r, c) - self.input_shape] == 3:
                    if player == 1:
                        self.p1Score += 1
                    else:
                        self.p2Score += 1
                    self.completed = True
                    return True
                else:
                    self.completed = False
                    return False
            except KeyError:
                try:
                    if self.completeBoxes[self.f(r, c) - self.input_shape] == 3 and r != 0:
                        if player == 1:
                            self.p1Score += 1
                        else:
                            self.p2Score += 1
                        self.completed = True
                        return True
                    else:
                        self.completed = False
                        return False
                except KeyError:
                    pass
        self.completed = False
        return False

    def update_boxes(self, action):

        """
        Update box completion structure based on current action
        :param action: chosen action
        :return:
        """

        # Unpack action
        r, c, o = action

        if o == 'h':
            try:
                self.completeBoxes[self.f(r, c)] += 1
                try:
                    self.completeBoxes[self.f(r, c) - self.input_shape] += 1
                except KeyError:
                    pass
            except KeyError:
                self.completeBoxes[self.f(r, c) - self.input_shape] += 1
        elif o == 'v':
            try:
                if c != self.input_shape:
                    self.completeBoxes[self.f(r, c)] += 1
                try:
                    if c != 0:
                        self.completeBoxes[self.f(r, c) - 1] += 1
                except KeyError:
                    pass
            except KeyError:
                self.completeBoxes[self.f(r, c) - 1] += 1

    def init_state(self, shape):

        """
        Initial state structure
        :param shape: state shape
        :return: state
        """

        s = [[{'v': 0, 'h': 0} for i in range(shape+1)] for j in range(shape+1)]
        snp = np.array(s)
        return snp

    def init_complete_boxes(self, input_shape):

        """
        Initialize box completion data structure
        :param input_shape: shape
        :return:
        """

        boxes = {}
        for i in range(1,input_shape*input_shape+1):
            boxes[i] = 0
        return boxes

    def change_state_dimension(self, state, d):

        """
        Convert state representation to tensor for model input
        :param state: state list(list(dict)))
        :param d: dimension
        :return:
        """

        new_state = np.zeros((d+1, d+1, 2))
        for i in range(len(state)):
            for j in range(len(state[i])):
                new_state[i][j][0] = state[i][j]['h']
                new_state[i][j][1] = state[i][j]['v']

        return new_state

    def f(self, r, c):
        return self.input_shape*r + (self.input_shape - (self.input_shape-c-1))

    def get_reward(self):

        """
        Calculate reward - step reward based on current game scores
        :return: reward value
        """

        if self.p1Score > self.p2Score and self.gameEnded is True:
            return 1
        elif self.p1Score < self.p2Score and self.gameEnded is True:
            return -1
        else:
            return 0

    def end_state(self, next_state):

        """
        Check whether or not the game is at its final state
        :param next_state: state
        :return: bool
        """

        for i in range(len(next_state)):
            for j in range(len(next_state[i])):
                if i == len(next_state)-1 and j == len(next_state)-1:
                    return True
                if next_state[i][j][0] == 0 and j < len(next_state)-1:
                    return False
                if next_state[i][j][1] == 0 and i < len(next_state)-1:
                    return False

    def reset_game(self, shape):

        """
        Method to reset game
        :param shape: board dimensions
        :return:
        """

        self.won = 0
        self.p1Score = 0
        self.p2Score = 0
        self.gameEnded = False
        self.game_replay_table = []
        self.state = self.init_state(shape)
        self.completeBoxes = self.init_complete_boxes(shape)


if __name__ == '__main__':

    sim = Simulation(4)
    sim.play()
    print("done")
