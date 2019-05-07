import random


class NaiveAgent:

    def __init__(self, input_shape):
        self.completed = False
        self.input_shape = input_shape

    def action(self, state, boxes):
        actions = self.generateMoves(state)
        for a in actions:
            if self.completeBox(a, boxes, self.input_shape):
                return a
        return random.choice(actions)

    def f(self, r, c):
        return (self.input_shape)*r + ((self.input_shape) -
                                       (self.input_shape-c-1))


    def generateMoves(self, feature_vector):
        moves = []
        for i in range(len(feature_vector)):
            for j in range(len(feature_vector)):
                if i == len(feature_vector) - 1 and j == len(
                        feature_vector) - 1:
                    return moves
                if feature_vector[i][j]['v'] == 0 and i < len(
                        feature_vector) - 1:
                    moves.append((i, j, 'v'))
                if feature_vector[i][j]['h'] == 0 and j < len(
                        feature_vector) - 1:
                    moves.append((i, j, 'h'))

    def completeBox(self, action, completeBoxes, input_shape):
        r, c, o = action
        if o == 'v':
            try:
                if completeBoxes[self.f(r, c)] == 3 \
                                            and c != input_shape:
                    self.completed = True
                    try:
                        if completeBoxes[self.f(r, c) - 1] == 3 and c != 0:
                            self.completed = True
                            return True
                        self.completed = True
                        return True
                    except KeyError:
                        pass
                    self.completed = True
                    return True
                elif completeBoxes[self.f(r, c) - 1] == 3 \
                                        and c != 0 \
                                        and c != input_shape:
                    self.completed = True
                    return True
                elif completeBoxes[self.f(r, c) - 1] == 3 \
                                            and c == input_shape:
                    self.completed = True
                    return True
                else:
                    self.completed = False
                    return False
            except KeyError:
                try:
                    if completeBoxes[self.f(r, c) - 1] == 3 and c != 0:
                        self.completed = True
                        return True
                    else:
                        self.completed = False
                        return False
                except KeyError:
                    pass
        elif o == 'h':
            try:
                if completeBoxes[self.f(r, c)] == 3:
                    self.completed = True
                    try:
                        if completeBoxes[self.f(r, c) -
                                              (input_shape)] == 3:
                            self.completed = True
                            return True
                    except KeyError:
                        pass
                    self.completed = True
                    return True
                elif completeBoxes[self.f(r, c) - (input_shape)] == 3:
                    self.completed = True
                    return True
                else:
                    self.completed = False
                    return False
            except KeyError:
                try:
                    if completeBoxes[self.f(r, c) -
                                          (input_shape)] == 3 and r != 0:
                        self.completed = True
                        return True
                    else:
                        self.completed = False
                        return False
                except KeyError:
                    pass
        self.completed = False
        return False