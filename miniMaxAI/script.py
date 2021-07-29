'''
#Min-Max Algorithm
The minimax algorithm is a decision-making algorithm that is used for finding the best move in a two player game.
It’s a recursive algorithm — it calls itself. In order for us to determine if making move A is a good idea, we need
to think about what our opponent would do if we made that move. We’d guess what our opponent would do by running the
minimax algorithm from our opponent’s point of view. As this process repeats, we can start to make a tree of these
hypothetical game states. We’ll eventually reach a point where the game is over — we’ll reach a leaf of the tree.
Either we won, our opponent won, or it was a tie. At this point, the recursion can stop. Because the game is over,
we no longer need to think about how our opponent would react if we reached this point of the game.
'''

from tic_tac_toe import *
from copy import deepcopy

start_board = [
        ["1", "2", "3"],
        ["4", "5", "6"],
        ["7", "8", "9"]
]

x_won = [
        ["X", "O", "3"],
        ["4", "X", "O"],
        ["7", "8", "X"]
]

o_won = [
        ["O", "X", "3"],
        ["O", "X", "X"],
        ["O", "8", "9"]
]

tie = [
        ["X", "X", "O"],
        ["O", "O", "X"],
        ["X", "O", "X"]
]

def game_is_over(board):
    return has_won(board, "X") or has_won(board, "O") or len(available_moves(board)) == 0

def evaluate_board(board):
    if has_won(board, "X"):
        return 1
    elif has_won(board, "O"):
        return -1
    else:
        return 0

def minimax(input_board, is_maximizing):
    if game_is_over(input_board):
        return [evaluate_board(input_board), ""]
    best_move = ""
    if is_maximizing == True:
        best_value = -float("Inf")
        symbol = "X"
    else:
        best_value = float("Inf")
        symbol = "O"
    for move in available_moves(input_board):
        new_board = deepcopy(input_board)
        select_space(new_board, move, symbol)
        hypothetical_value = minimax(new_board, not is_maximizing)[0]
        if is_maximizing == True and hypothetical_value > best_value:
            best_value = hypothetical_value
            best_move = move
        if is_maximizing == False and hypothetical_value < best_value:
            best_value = hypothetical_value
            best_move = move
    return [best_value, best_move]


#AI vs AI Game

my_board = [
        ["1", "2", "3"],
        ["4", "5", "6"],
        ["7", "8", "9"]
]

while not game_is_over(my_board):
    select_space(my_board, minimax(my_board, True)[1], "X")
    print_board(my_board)
    if not game_is_over(my_board):
        choice = input("Select a move:\n")
        try:
            move = int(choice)
        except ValueError:
            print("Wrong choice!")
        select_space(my_board, move, "O")
        print_board(my_board)
