from game import Game, Move
import numpy as np
from copy import deepcopy
import random


def get_possible_moves2(board, current_player_idx):
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

def get_possible_moves(board, current_player_idx):
    possible_moves = []
    board_size = len(board)
    SIDES = [(0, 0), (0, board_size - 1), (board_size - 1, 0), (board_size - 1, board_size - 1)]

    for x in range(board_size):
        for y in range(board_size):
            if not is_valid_move(board, x, y, current_player_idx):
                continue

            # Add moves based on position (corners, sides)
            if (x, y) in SIDES:  # Corner cases with specific allowed moves
                if (x, y) == (0, 0):  # Top left
                    possible_moves += [((x, y), Move.BOTTOM), ((x, y), Move.RIGHT)]
                elif (x, y) == (0, board_size - 1):  # Top right
                    possible_moves += [((x, y), Move.BOTTOM), ((x, y), Move.LEFT)]
                elif (x, y) == (board_size - 1, 0):  # Bottom left
                    possible_moves += [((x, y), Move.TOP), ((x, y), Move.RIGHT)]
                elif (x, y) == (board_size - 1, board_size - 1):  # Bottom right
                    possible_moves += [((x, y), Move.TOP), ((x, y), Move.LEFT)]
            else:  # Edge but not corner
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
    is_on_border = x == 0 or x == board_size - 1 or y == 0 or y == board_size - 1
    can_be_moved_by_player = board[x, y] < 0 or board[x, y] == current_player_idx
    return is_on_border and can_be_moved_by_player

def apply_move(board, move, player_id):
    from_pos, slide = move

    # swap the coordinates (?)
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

def count_aligned(board, player_id, exact_count=4):
        max_elements = 0
        board_size = len(board)
        otk = False

        # check rows, cols and diags
        for i in range(board_size):
            row_count = sum(board[i, :] == player_id)
            col_count = sum(board[:, i] == player_id)
            max_elements = max(max_elements, row_count, col_count)
            if row_count == exact_count or col_count == exact_count:
                otk = True

        # diags
        diag_count = sum(board[i, i] == player_id for i in range(board_size))
        anti_diag_count = sum(board[i, board_size - i - 1] == player_id for i in range(board_size))
        max_elements = max(max_elements, diag_count, anti_diag_count)
        if diag_count == exact_count or anti_diag_count == exact_count:
            otk = True

        return max_elements, otk

def normalize_state_binary(board, player_index):
    player_bits = np.zeros(25, dtype=np.uint8)
    opponent_bits = np.zeros(25, dtype=np.uint8)
    
    opponent_index = 1 - player_index
    
    for i in range(5):  # per una board 5x5
        for j in range(5):
            if board[i, j] == player_index:
                player_bits[i * 5 + j] = 1
            elif board[i, j] == opponent_index:
                opponent_bits[i * 5 + j] = 1
    
    # Concatena i bit del player, i separatori e i bit dell'opponente
    state_bits = np.concatenate(([0]*7, player_bits, [0]*7, opponent_bits))
    
    return state_bits

def normalize_state_simple(board, player_index):
    state = np.zeros(board.shape[0] * board.shape[1], dtype=np.float32)
    
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j] == player_index:
                state[i * board.shape[1] + j] = 1  # player
            elif board[i, j] == -1:
                state[i * board.shape[1] + j] = 0  # vuota
            else:
                state[i * board.shape[1] + j] = -1  # opponent
    
    return state

def check_winner(board):
    """
    Controlla se c'è un vincitore
    """
    # righe e colonne
    for i in range(board.shape[0]):
        if np.all(board[i, :] == 0) or np.all(board[:, i] == 0):
            return 0  # 0 vince
        if np.all(board[i, :] == 1) or np.all(board[:, i] == 1):
            return 1  # 1 vince
    
    # diagonali
    if np.all(np.diag(board) == 0) or np.all(np.diag(np.fliplr(board)) == 0):
        return 0  # 0 vince
    if np.all(np.diag(board) == 1) or np.all(np.diag(np.fliplr(board)) == 1):
        return 1  # 1 vince

    # pareggio
    if not np.any(board == -1):  # Non ci sono mosse possibili
        return -2
    
    # Nessun vincitore
    return -1

def print_board(board):
    for row in board:
        print(" ".join(str(cell) for cell in row))
    print("\n")
