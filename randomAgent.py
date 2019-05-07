import random


class RandomAgent:

    def __init__(self, state):
        self.state = state

    def action(self,state, boxes):
        # Random move
        free_lines = []
        for ri in range(len(state)):
            row = state[ri]
            for ci in range(len(row)):
                cell = row[ci]
                if ri < (len(state) - 1) and cell["v"] == 0:
                    free_lines.append((ri, ci, "v"))
                if ci < (len(row) - 1) and cell["h"] == 0:
                    free_lines.append((ri, ci, "h"))
        if len(free_lines) == 0:
            # Board full
            return None
        movei = random.randint(0, len(free_lines) - 1)
        r, c, o = free_lines[movei]
        return r, c, o