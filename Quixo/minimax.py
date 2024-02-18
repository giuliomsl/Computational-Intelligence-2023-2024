import random
from tqdm import tqdm
from utils import *
from game import Game, Move, Player


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game) -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        # print("RandomPlayer from_pos:", from_pos)  # debug
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move

class MiniMaxPlayer(Player):
    def __init__(self, depth: int, player_index) -> None:
        super().__init__()
        self.depth = depth
        self.player_index = player_index

    def evaluate_game_state(self, board):
        player_score, player_4_in_row = count_aligned(board, self.player_index, 4)
        opponent_score, opponent_4_in_row = count_aligned(board, 1 - self.player_index, 4)

        # evaluation of important pieces (corners) and alligned pieces

        strategic_value = 0
        corners = [(0, 0), (0, len(board)-1), (len(board)-1, 0), (len(board)-1, len(board)-1)]
        for x, y in corners:
            if board[x, y] == self.player_index:
                strategic_value += 1
            elif board[x, y] == 1 - self.player_index:
                strategic_value -= 1

        # critical situations
        if opponent_4_in_row:
            return -10  # avoid opponent's win
        if player_4_in_row:
            return 10   # win the game

        return player_score - opponent_score + strategic_value


    def minimax(self, board, depth, alpha, beta, maximizing_player):

        if depth == 0:
            return self.evaluate_game_state(board)

        if maximizing_player:
            max_eval = -float('inf')
            for move in get_possible_moves(board, self.player_index):
                # apply the move and compute new board state
                new_board = apply_move(board, move, self.player_index)
                eval = self.minimax(new_board, depth - 1, alpha, beta, False)
                # print(f"Valutazione mossa {move}: {eval}")  # Debugging
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in get_possible_moves(board, 1 - self.player_index):
                # apply the move and compute new board state
                new_board = apply_move(board, move, 1 - self.player_index)
                eval = self.minimax(new_board, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
        

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        best_score = -float('inf')
        best_action = None  # tuple (from_pos, move)

        for action in get_possible_moves(game.get_board(), self.player_index):
            # apply the move to obtain the new board 
            new_board = apply_move(game.get_board(), action, self.player_index)
            score = self.minimax(new_board, self.depth, -float('inf'), float('inf'), True)
            if score > best_score:
                best_score = score
                best_action = action

        # print(f"MiniMaxPlayer best_action: from_pos={best_action[0]}, move={best_action[1]}")
        # print_board(new_board)
        return (best_action[0][1], best_action[0][0]), best_action[1] if best_action is not None else ((0, 0), Move.TOP)

if __name__ == "__main__":
    wins = 0
    draws = 0
    losses = 0
    games = 100
    for i in tqdm(range(games)):
        game = Game()
        player1 = MiniMaxPlayer(3, 1)
        player2 = RandomPlayer()
        winner = game.play(player1, player2)
        
        if winner == player1.player_index:
            game_result = "win"
            wins += 1
        elif winner == 2:  
            game_result = "draw"
            draws += 1 
        else:
            game_result = "lose"
            losses += 1
    print(f"Percentuale vittorie: {wins / games * 100}%")
    print(f"Percentuale pareggi: {draws / games * 100}%") 
    print(f"Percentuale sconfitte: {losses / games * 100}%")
    
    # depth = 5 has a great win rate (between 75% and 95%), almost 20 seconds per game

#   depth = 5 vs. Random
# 100%|███████████████████████████████████████████████████████████████████████████| 100/100 [31:16<00:00, 18.76s/it]
# Percentuale vittorie: 94.0
# Percentuale pareggi: 0.0%
# Percentuale sconfitte: 6.0%
    
#   depth = 3 vs. Random
# 100%|███████████████████████████████████████████████████████████████████████████| 100/100 [03:04<00:00,  1.85s/it]
# Percentuale vittorie: 86.0%
# Percentuale pareggi: 0.0%
# Percentuale sconfitte: 14.0%
