# Tic Tac Toe using Minimax Algorithm
# Experiment 12

import math

# Board
board = [' ' for _ in range(9)]

# Print board
def print_board():
    print()
    for i in range(0, 9, 3):
        print(board[i], "|", board[i+1], "|", board[i+2])
    print()

# Check winner
def check_winner(player):
    win_conditions = [
        [0,1,2],[3,4,5],[6,7,8],
        [0,3,6],[1,4,7],[2,5,8],
        [0,4,8],[2,4,6]
    ]
    for cond in win_conditions:
        if all(board[i] == player for i in cond):
            return True
    return False

# Check draw
def is_draw():
    return ' ' not in board

# Minimax algorithm
def minimax(is_maximizing):
    if check_winner('O'):
        return 1
    if check_winner('X'):
        return -1
    if is_draw():
        return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'O'
                score = minimax(False)
                board[i] = ' '
                best_score = max(best_score, score)
        return best_score
    else:
        best_score = math.inf
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'X'
                score = minimax(True)
                board[i] = ' '
                best_score = min(best_score, score)
        return best_score

# Best move for AI
def best_move():
    best_score = -math.inf
    move = 0
    for i in range(9):
        if board[i] == ' ':
            board[i] = 'O'
            score = minimax(False)
            board[i] = ' '
            if score > best_score:
                best_score = score
                move = i
    board[move] = 'O'

# Game loop
def play_game():
    print("Tic Tac Toe Game")
    print("You are X | Computer is O")

    while True:
        print_board()
        move = int(input("Enter position (1-9): ")) - 1
        if board[move] != ' ':
            print("Invalid move!")
            continue
        board[move] = 'X'

        if check_winner('X'):
            print_board()
            print("You win!")
            break
        if is_draw():
            print_board()
            print("Draw!")
            break

        best_move()

        if check_winner('O'):
            print_board()
            print("Computer wins!")
            break
        if is_draw():
            print_board()
            print("Draw!")
            break

play_game()
