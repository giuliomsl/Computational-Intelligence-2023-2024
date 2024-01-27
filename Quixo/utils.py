from game import Game, Move
import numpy as np
from copy import deepcopy
import random

def get_possible_moves(board, current_player_idx):
    possible_moves = []
    board_size = len(board)

    for x in range(board_size):
        for y in range(board_size):
            if not is_valid_move(board, x, y, current_player_idx):
                continue

            if x == 0:
                possible_moves.append(((x, y), Move.BOTTOM))
            elif x == board_size - 1:
                possible_moves.append(((x, y), Move.TOP))

            if y == 0:
                possible_moves.append(((x, y), Move.RIGHT))
            elif y == board_size - 1:
                possible_moves.append(((x, y), Move.LEFT))

    return list(set(possible_moves))

def is_valid_move(board, x, y, current_player_idx):
    board_size = len(board)
    # check if the piece is on the border
    is_on_border = x == 0 or x == board_size - 1 or y == 0 or y == board_size - 1
    # check if the piece is choosable
    can_be_moved_by_player = board[x, y] < 0 or board[x, y] == current_player_idx

    return is_on_border and can_be_moved_by_player


def apply_move(board, move, player_id):
    from_pos, slide = move

    # swap the coordinates
    from_pos = (from_pos[1], from_pos[0])

    # take control of the piece
    board[from_pos] = player_id
    piece = board[from_pos]

    # slide all the pieces
    if slide == Move.LEFT:
        for i in range(from_pos[1], 0, -1):
            board[from_pos[0], i] = board[from_pos[0], i - 1]
        board[from_pos[0], 0] = piece

    elif slide == Move.RIGHT:
        for i in range(from_pos[1], len(board[0]) - 1):
            board[from_pos[0], i] = board[from_pos[0], i + 1]
        board[from_pos[0], len(board[0]) - 1] = piece

    elif slide == Move.TOP:
        for i in range(from_pos[0], 0, -1):
            board[i, from_pos[1]] = board[i - 1, from_pos[1]]
        board[0, from_pos[1]] = piece

    elif slide == Move.BOTTOM:
        for i in range(from_pos[0], len(board) - 1):
            board[i, from_pos[1]] = board[i + 1, from_pos[1]]
        board[len(board) - 1, from_pos[1]] = piece

    return board

def print_board(board):
    for row in board:
        print(" ".join(str(cell) for cell in row))
    print("\n")