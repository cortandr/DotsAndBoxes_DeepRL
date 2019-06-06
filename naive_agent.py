import random


class NaiveAgent:

    def __init__(self, input_shape):
        self.completed = False
        self.input_shape = input_shape

    def action(self, state, boxes):

        """
        Choose action
        :param state: current state
        :param boxes: current box completion data structure
        :return:
        """
        actions = self.generate_moves(state)
        for a in actions:
            if self.complete_box(a, boxes, self.input_shape):
                return a
        return random.choice(actions)

    def f(self, r, c):

        """
        Method used to convert row and col coords to box coordinate used in
        completed boxes data structure
        :param r:
        :param c:
        :return:
        """
        return self.input_shape*r + (self.input_shape -
                                       (self.input_shape-c-1))

    def generate_moves(self, feature_vector):

        """
        Generate all allowed moves
        :param feature_vector: state
        :return: list of actions
        """
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

    def complete_box(self, action, complete_boxes, input_shape):

        """
        Method used to check for completed boxes after application of an action
        If also assigns points to the player based on completion
        :param action: chosen action
        :param complete_boxes: boxes completion data structure
        :param input_shape: input shape
        :return: bool
        """

        # Unpack action
        r, c, o = action

        if o == 'v':
            try:
                if complete_boxes[self.f(r, c)] == 3 and c != input_shape:

                    # Assign completion of box
                    self.completed = True

                    try:
                        # Check for box to the left
                        if complete_boxes[self.f(r, c) - 1] == 3 and c != 0:

                            # Assign completion of box
                            self.completed = True
                            return True

                        self.completed = True
                        return True
                    except KeyError:
                        pass

                    self.completed = True
                    return True

                elif complete_boxes[self.f(r, c) - 1] == 3 and c != 0 and c != input_shape:
                    self.completed = True
                    return True
                elif complete_boxes[self.f(r, c) - 1] == 3 and c == input_shape:
                    self.completed = True
                    return True
                else:
                    self.completed = False
                    return False
            except KeyError:
                try:
                    if complete_boxes[self.f(r, c) - 1] == 3 and c != 0:
                        self.completed = True
                        return True
                    else:
                        self.completed = False
                        return False
                except KeyError:
                    pass
        elif o == 'h':
            try:
                if complete_boxes[self.f(r, c)] == 3:
                    self.completed = True
                    try:
                        if complete_boxes[self.f(r, c) - input_shape] == 3:
                            self.completed = True
                            return True
                    except KeyError:
                        pass
                    self.completed = True
                    return True
                elif complete_boxes[self.f(r, c) - input_shape] == 3:
                    self.completed = True
                    return True
                else:
                    self.completed = False
                    return False
            except KeyError:
                try:
                    if complete_boxes[self.f(r, c) - input_shape] == 3 and r != 0:
                        self.completed = True
                        return True
                    else:
                        self.completed = False
                        return False
                except KeyError:
                    pass
        self.completed = False
        return False
