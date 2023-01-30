import random
import copy
import math


class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        s = -math.inf
        move = []
        temp_move = None
        temp_source = None
        successors = self.succ_states(state, self.my_piece)
        drop_phase = self.get_drop_phase(state)
        if drop_phase:
            for successor in successors:
                temp_state = copy.deepcopy(state)
                temp_state[successor[0]][successor[1]] = self.my_piece
                temp_score = self.max_value(temp_state, 0, -math.inf, math.inf)
                if (temp_score > s):
                    s = temp_score
                    temp_move = (successor[0], successor[1])
        else:
            for successor in successors:
                temp_state = copy.deepcopy(state)
                temp_state[successor[1][0]][successor[1][1]] = ' '
                temp_state[successor[0][0]][successor[0][1]] = self.my_piece
                temp_score = self.max_value(temp_state, 0, -math.inf, math.inf)
                if (temp_score > s):
                    s = temp_score
                    temp_move = (successor[0][0], successor[0][1])
                    temp_source = (successor[1][0], successor[1][1])

        if temp_source is not None:
            move.append(temp_move)
            move.append(temp_source)
        else:
            move.append(temp_move)
        # ensure the destination (row,col) tuple is at the beginning of the move list
        # select an unoccupied space randomly
        # TODO: implement a minimax algorithm to play better
        # (row, col) = (random.randint(0,4), random.randint(0,4))
        # while not state[row][col] == ' ':
        #     (row, col) = (random.randint(0,4), random.randint(0,4))
        return move


    def get_drop_phase(self, state):
        drop_phase = True
        count_non_space = 0
        for i in state:
            for j in i:
                if j != ' ':
                    count_non_space += 1
                    if count_non_space == 8:
                        drop_phase = False
                        break
        return drop_phase

    def get_adjacent_states(self, i, j):
        adjacent_states = []
        possible_moves = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1), (i - 1, j - 1), (i + 1, j - 1), (i - 1, j + 1), (i + 1, j + 1)]
        for k in possible_moves:
            if 0 <= k[0] < 5 and 0 <= k[1] < 5:
                adjacent_states.append(k)
        return adjacent_states

    def succ_states(self, state, my_piece):
        drop_phase = self.get_drop_phase(state)
        successors = []
        if drop_phase:
            for row in range(5):
                for column in range(5):
                    if state[row][column] == ' ':
                        successors.append((row, column))
        else:
            for i in range(5):
                for j in range(5):
                    if state[i][j] == my_piece:
                        adjacent_states = self.get_adjacent_states(i, j)
                        for st in adjacent_states:
                            if state[st[0]][st[1]] == ' ':
                                successor = list([(st[0], st[1]), (i, j)])
                                successors.append(successor)
        return successors

    def heuristic_game_value(self, state):
        terminal_state = self.game_value(state)
        if terminal_state != 0:
            return terminal_state
        w = [
            [0, 1, 0, 1, 0],
            [1, 2, 2, 2, 1],
            [0, 2, 3, 2, 0],
            [1, 2, 2, 2, 1],
            [0, 1, 0, 1, 0]
        ]
        curr_player_score = 0
        opp_score = 0
        for i in range(5):
            for j in range(5):
                if (state[i][j] == self.my_piece):
                    curr_player_score = curr_player_score + w[i][j]
                elif state[i][j] == self.opp:
                    opp_score = opp_score + w[i][j]
        return curr_player_score - opp_score

    def max_value(self, state, depth, alpha, beta):
        if depth > 1:
            return self.heuristic_game_value(state)
        if abs(self.game_value(state)) == 1:
            return self.game_value(state)
        drop_phase = self.get_drop_phase(state)
        successors = self.succ_states(state, self.my_piece)
        if drop_phase:
            for successor in successors:
                temp_state = copy.deepcopy(state)
                temp_state[successor[0]][successor[1]] = self.my_piece
                alpha = max(alpha, self.min_value(temp_state, depth + 1, alpha, beta))
        else:
            for successor in successors:
                temp_state = copy.deepcopy(state)
                temp_state[successor[1][0]][successor[1][1]] = ' '
                temp_state[successor[0][0]][successor[0][1]] = self.my_piece
                alpha = max(alpha, self.min_value(temp_state, depth + 1, alpha, beta))
        if alpha >= beta:
            return beta
        return alpha

    def min_value(self, state, depth, alpha, beta):
        if depth > 1:
            return self.heuristic_game_value(state)
        if abs(self.game_value(state)) == 1:
            return self.game_value(state)
        drop_phase = self.get_drop_phase(state)
        successors = self.succ_states(state, self.opp)
        if drop_phase:
            for successor in successors:
                temp_state = copy.deepcopy(state)
                temp_state[successor[0]][successor[1]] = self.opp
                beta = min(beta, self.max_value(temp_state, depth + 1, alpha, beta))
        else:
            for successor in successors:
                temp_state = copy.deepcopy(state)
                temp_state[successor[1][0]][successor[1][1]] = ' '
                temp_state[successor[0][0]][successor[0][1]] = self.opp
                beta = min(beta, self.max_value(temp_state, depth + 1, alpha, beta))
        if alpha >= beta:
            return alpha
        return beta


    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row) + ": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i + 1] == row[i + 2] == row[i + 3]:
                    return 1 if row[i] == self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i + 1][col] == state[i + 2][col] == state[i + 3][col]:
                    return 1 if state[i][col] == self.my_piece else -1
        '''
        # TODO: check \ diagonal wins
        # think of a better way to do it
        if state[0][0] != ' ' and state[0][0] == state[1][1] == state[2][2] == state[3][3]:
            return 1 if state[0][0] == self.my_piece else -1
        elif state[0][1] != ' ' and state[0][1] == state[1][2] == state[2][3] == state[3][4]:
            return 1 if state[0][1] == self.my_piece else -1
        elif state[1][0] != ' ' and state[1][0] == state[2][1] == state[3][2] == state[4][3]:
            return 1 if state[1][0] == self.my_piece else -1
        elif state[1][1] != ' ' and state[1][1] == state[2][2] == state[3][3] == state[4][4]:
            return 1 if state[1][1] == self.my_piece else -1

        # TODO: check / diagonal wins
        if state[0][3] != ' ' and state[0][3] == state[1][2] == state[2][1] == state[3][0]:
            return 1 if state[0][3] == self.my_piece else -1
        elif state[0][4] != ' ' and state[0][4] == state[1][3] == state[2][2] == state[3][1]:
            return 1 if state[0][4] == self.my_piece else -1
        elif state[1][3] != ' ' and state[1][3] == state[2][2] == state[3][1] == state[4][0]:
            return 1 if state[1][3] == self.my_piece else -1
        elif state[1][4] != ' ' and state[1][4] == state[2][3] == state[3][2] == state[4][1]:
            return 1 if state[1][4] == self.my_piece else -1
        '''
        # TODO: check \ diagonal wins
        for col in range(2):
            for i in range(3, 5):
                if state[i][col] != ' ' and state[i][col] == state[i - 1][col + 1] == state[i - 2][col + 2] == state[i - 3][col + 3]:
                    return 1 if state[i][col] == self.my_piece else -1

        # TODO: check \ diagonal wins
        for col in range(2):
            for i in range(0, 2):
                if state[i][col] != ' ' and state[i][col] == state[i + 1][col + 1] == state[i + 2][col + 2] == state[i + 3][col + 3]:
                    return 1 if state[i][col] == self.my_piece else -1

        # TODO: check box wins
        for col in range(4):
            for i in range(4):
                if state[i][col] != ' ' and state[i][col] == state[i][col + 1] == state[i + 1][col] == state[i + 1][col + 1]:
                    return 1 if state[i][col] == self.my_piece else -1

        return 0  # no winner yet


############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece + " moved at " + chr(move[0][1] + ord("A")) + str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0]) - ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece + " moved from " + chr(move[1][1] + ord("A")) + str(move[1][0]))
            print("  to " + chr(move[0][1] + ord("A")) + str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0]) - ord("A")),
                                      (int(move_from[1]), ord(move_from[0]) - ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()