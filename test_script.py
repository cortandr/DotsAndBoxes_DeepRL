import numpy as np
import copy
from naive_agent import NaiveAgent
from DQN import DQN
from randomAgent import RandomAgent


class Test:

    def __init__(self, input_shape, path, trained_model):

        self.trained_model = trained_model
        self.player1 = None
        self.current_turn = 2
        self.player2 = None
        self.p1Score = 0
        self.p2Score = 0
        self.prevState = None
        self.state = self.initState(input_shape)
        self.simNo = 2000
        self.gameEnded = False
        self.action = None
        self.won = 0

        self.completeBoxes = self.initCompleteBoxes(input_shape)
        self.completed = False
        self.path = path
        self.path2 = "./saved_model3x3.FINAL"
        self.learnerWonGames = 0
        self.learnerTiedGames = 0
        self.learnerLostGames = 0
        self.wins_random = 0
        self.tie_random = 0
        self.lost_random = 0
        self.wins_naive = 0
        self.input_shape = input_shape

    def play(self):
        # Initialize RL agent
        self.player1 = DQN(self.input_shape, False)
        # Load trained model
        self.player1.load_model(self.path + "/model")
        #Initialize Opponent
        self.player2 = RandomAgent(self.input_shape)
        # self.player2 = DQN(self.input_shape, False)
        # self.player2.load_model(self.path2 + "/model")
        i = 0
        while i < self.simNo:
            i += 1
            # PLay game
            while self.gameEnded == False:
                # Adapt state for network's input
                feature_vector = self.change_state_dimension(
                    copy.deepcopy(self.state),
                    self.input_shape)
                if self.current_turn == 1:
                    # Choose best action
                    self.action = self.player1.predict(feature_vector)
                    self.applyAction(self.action, 1)
                    # Decide who will play next turn
                    if self.completed is True:
                        self.current_turn = 1
                    else:
                        self.current_turn = 2
                else:
                    a = self.player2.action(self.state, self.completeBoxes)
                    #a = self.player2.predict(feature_vector)
                    self.applyAction(a, 2)
                    if self.completed is True:
                        self.current_turn = 2
                    else:
                        self.current_turn = 1

            # Switch the starting player every 500 games
            if i >= i/4:
                self.current_turn = 2
            else:
                self.current_turn = 1

            if i == self.simNo/2:
                self.player2 = NaiveAgent(self.input_shape)
                self.wins_random = copy.deepcopy(self.learnerWonGames)
                self.tie_random = copy.deepcopy(self.learnerTiedGames)
                self.lost_random = copy.deepcopy(self.learnerLostGames)
                self.learnerWonGames = 0
                self.learnerTiedGames = 0
                self.learnerLostGames = 0

            self.resetGame(self.input_shape)

        print("Model Tested on a {}x{} board".format(self.input_shape, self.input_shape))
        print("--------------------------------------------")
        print("Win rate RL agent vs Random agent: " +
              "{0:.1f}".format((self.wins_random/(self.simNo/2))*100) + "%")
        print("Draw rate RL agent vs Random agent: " +
              "{0:.1f}".format(
                  (self.tie_random / (self.simNo / 2)) * 100) + "%")
        print("Loss rate RL agent vs Random agent: " +
              "{0:.1f}".format(
                  (self.lost_random / (self.simNo / 2)) * 100) + "%")
        print(" ")
        print("Win rate RL agent vs Naive agent: " +
              "{0:.1f}".format(
                  (self.learnerWonGames / (self.simNo / 2)) * 100) + "%")
        print("Draw rate RL agent vs Naive agent: " +
              "{0:.1f}".format(
                  (self.learnerTiedGames / (self.simNo / 2)) * 100) + "%")
        print("Loss rate RL agent vs Naive agent: " +
              "{0:.1f}".format(
                  (self.learnerLostGames / (self.simNo / 2)) * 100) + "%")
        print("--------------------------------------------")
        print(" ")
        # print("Tested on a {}x{} board".format(self.input_shape,
        #                                              self.input_shape))
        # print("--------------------------------------------")
        # print("Win rate 4x4 agent vs 3x3 agent: " +
        #       "{0:.1f}".format((self.learnerWonGames/(self.simNo))*100) + "%")
        # print("Draw rate 4x4 agent vs 3x3 agent: " +
        #       "{0:.1f}".format(
        #           (self.learnerTiedGames / (self.simNo)) * 100) + "%")
        # print("Loss rate 4x4 agent vs 3x3 agent: " +
        #       "{0:.1f}".format(
        #           (self.learnerLostGames / (self.simNo)) * 100) + "%")
        # print("--------------------------------------------")


    def applyAction(self, action, player):
        r, c, o = action
        # Apply action
        self.state[r][c][o] = player
        # Update counters
        if self.completeBox(action, player) and self.gameEnded is False:
            if self.p1Score + self.p2Score >= self.input_shape**2:
                if self.p1Score > self.p2Score:
                    self.won = 1
                    self.learnerWonGames += 1
                elif self.p1Score == self.p2Score:
                    self.won = 0
                    self.learnerTiedGames += 1
                else:
                    self.won = 2
                    self.learnerLostGames += 1
                self.gameEnded = True
        #update the number of segments per box
        self.updateBoxes(action)
        return

    def generateMoves(self, feature_vector):
        moves = []
        for i in range(self.input_shape + 1):
            for j in range(self.input_shape + 1):
                if i == self.input_shape and j == self.input_shape:
                    return moves
                if feature_vector[i][j][1] == 0 and i < len(feature_vector) - 1:
                    moves.append((i, j, 'v'))
                if feature_vector[i][j][0] == 0 and j < len(feature_vector) - 1:
                    moves.append((i, j, 'h'))

    def completeBox(self, action, player):
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
                        if self.completeBoxes[self.f(r, c) -
                                              (self.input_shape)] == 3:
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
                elif self.completeBoxes[self.f(r, c) - (self.input_shape)] == 3:
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
                    if self.completeBoxes[self.f(r, c) -
                                          (self.input_shape)] == 3 and r != 0:
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

    def updateBoxes(self, action):
        r, c, o = action
        if o == 'h':
            try:
                self.completeBoxes[self.f(r, c)] += 1
                try:
                    self.completeBoxes[self.f(r, c) - (self.input_shape)] += 1
                except KeyError:
                    pass
            except KeyError:
                self.completeBoxes[self.f(r, c) - (self.input_shape)] += 1
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
        return

    def initState(self, shape):
        s = [[{'v': 0, 'h': 0} for i in range(shape + 1)] for j in
             range(shape + 1)]
        snp = np.array(s)
        return snp

    def initCompleteBoxes(self, input_shape):
        boxes = {}
        for i in range(1, input_shape * input_shape + 1):
            boxes[i] = 0
        return boxes

    def change_state_dimension(self, state, d):

        new_state = np.zeros((d + 1, d + 1, 2))
        for i in range(len(state)):
            for j in range(len(state[i])):
                new_state[i][j][0] = state[i][j]['h']
                new_state[i][j][1] = state[i][j]['v']

        return new_state

    def f(self, r, c):
        return (self.input_shape) * r + ((self.input_shape) -
                                         (self.input_shape - c - 1))


    def end_state(self, next_state):
        for i in range(len(next_state)):
            for j in range(len(next_state[i])):
                if i == len(next_state) - 1 and j == len(next_state) - 1:
                    return True
                if next_state[i][j][0] == 0 and j < len(next_state) - 1:
                    return False
                if next_state[i][j][1] == 0 and i < len(next_state) - 1:
                    return False

    def resetGame(self, shape):
        self.won = 0
        self.p1Score = 0
        self.p2Score = 0
        self.state = self.initState(shape)
        self.gameEnded = False
        self.completeBoxes = self.initCompleteBoxes(shape)


path3x3 = "./saved_model3x3.FINAL"
path4x4 = "./saved_model4x4.FINAL"

# MODEL TRAINED ON 3X3
print("TEST RAN WITH MODEL TRAINED ON A {}x{} BOARD".format(3, 3))
print(" ")
s = Test(2, path3x3, 3)
s.play()
s = Test(3, path3x3, 3)
s.play()
s = Test(4, path3x3, 3)
s.play()
s = Test(5, path3x3, 3)
s.play()


# MODEL TRAINED ON 4X4
print("TEST RAN WITH MODEL TRAINED ON A {}x{} BOARD".format(4, 4))
print(" ")
s = Test(2, path4x4, 4)
s.play()
s = Test(3, path4x4, 4)
s.play()
s = Test(4, path4x4, 4)
s.play()
s = Test(5, path4x4, 4)
s.play()

# # MODEL 4x4 AGAINST 3X3
# print("TEST RAN WITH BOTH MODELS AGAINST EACH OTHER")
# print(" ")
# s = Test(2, path4x4, 4)
# s.play()
# s = Test(3, path4x4, 4)
# s.play()
# s = Test(4, path4x4, 4)
# s.play()
# s = Test(5, path4x4, 4)
# s.play()


