import random
import pickle
from tqdm import tqdm
from utils import *
from copy import deepcopy
from minimax import MiniMaxPlayer
from game import Game, Move, Player


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game) -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])

        return from_pos, move
    
class QLearningPlayer(Player):
    def __init__(self, player_index, alpha=0.2, gamma=0.9, epsilon=0.2, epsilon_decay=1, epsilon_min=0, preload=True):
        super().__init__()
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration probability
        self.epsilon_decay = epsilon_decay  # Epsilon decay rate
        self.epsilon_min = epsilon_min  # Minimum epsilon
        self.q_table = {}  # Initialize Q-table
        self.moves_history = []  # Track moves
        self.player_index = player_index
        self.win_count = 0
        self.draw_count = 0
        self.loss_count = 0

        if preload:
            self.q_table = self.load_q_table()

    def load_q_table(self):
        try:
            with open("q_table.pickle", "rb") as f:
                print("File Q-table caricato.")
                return pickle.load(f)
        except FileNotFoundError:
            print("File Q-table non trovato, inizializzazione di una nuova Q-table.")
            return {}
        
    def save_q_table(self):
        try:
            with open("q_table.pickle", "wb") as f:
                pickle.dump(self.q_table, f)
            print("Q-table salvata con successo.")
        except Exception as e:
            print(f"Errore nel salvataggio della Q-table: {e}")
        
    def state_representation(self, board):
        return str(board.reshape(-1))

    def update_q_table(self, state, action, reward, next_max):
        if state not in self.q_table:
            self.q_table[state] = {}
        
        action_key = self.move_to_key(action)
        if action_key not in self.q_table[state]:
            self.q_table[state][action_key] = 0  # Inizializza se non presente

        # Aggiorna Q usando next_max, ovvero il massimo valore Q per il prossimo stato
        self.q_table[state][action_key] += self.alpha * (reward + self.gamma * next_max - self.q_table[state][action_key])

    '''
    def adjust_epsilon_dynamically(self):
        # Riduce epsilon più lentamente se l'agente sta vincendo o numero di pareggi alto
        if self.win_count > self.loss_count or self.draw_count > (self.win_count + self.loss_count):
            self.epsilon *= (self.epsilon_decay ** 0.5)  # Riduzione più lenta
        elif self.loss_count >= self.win_count:
            # Aumenta epsilon leggermente se ci sono molte sconfitte, esploraz. più ampia
            self.epsilon = min(self.epsilon / (self.epsilon_decay ** 0.5), 1.0)
        
        self.epsilon = max(self.epsilon, self.epsilon_min)
    '''
        
    def make_move(self, game) -> tuple[tuple[int, int], Move]:
        board = game.get_board()
        state = self.state_representation(board)

        possible_moves = get_possible_moves(board, self.player_index)

        if random.random() < self.epsilon or state not in self.q_table:
            selected_move = random.choice(possible_moves)
        else:
            max_q_value = float("-inf")
            selected_action_key = None
            for move in possible_moves:
                action_key = self.move_to_key(move)
                q_value = self.q_table.get(state, {}).get(action_key, float("-inf"))
                if q_value > max_q_value:
                    max_q_value = q_value
                    selected_action_key = action_key

            # Se nessuna mossa ha Q superiore, scelta casuale tra le mosse
            if selected_action_key is None:
                selected_move = random.choice(possible_moves)
            else:
                selected_move = self.key_to_move(selected_action_key)

        # Simula mossa selezionata per valutare il suo effetto
        simulated_board = apply_move(deepcopy(board), selected_move, self.player_index)
        simulated_state = self.state_representation(simulated_board)
        reward = self.evaluate_move_effect(game, selected_move)

        # Calcola il miglior Q per il prossimo stato simulato
        next_max = max(self.q_table[simulated_state].values(), default=0) if simulated_state in self.q_table else 0
        self.update_q_table(state, selected_move, reward, next_max)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        #print(f"Epsilon: {self.epsilon}")

        self.moves_history.append((state, selected_move, simulated_state))

        #print("LOOP")
        #print(f"{board}")
        #print(f"Selected Move: {selected_move[0][0], selected_move[0][1]}")
        #print(f"Possible Moves {possible_moves}")
        selected_move = ((selected_move[0][1], selected_move[0][0]), selected_move[1])

        return selected_move


    def move_to_key(self, move):
        # Converte la mossa in una chiave utilizzabile nella Q-table
        return (move[0], move[1].value)

    def key_to_move(self, key):
        # Converte una chiave della Q-table in una mossa
        return (key[0], Move(key[1]))
    
    def learn(self, game, move, reward):
        board = game.get_board()
        state = self.state_representation(board)
        next_state = deepcopy(board) 
        apply_move(next_state, move, self.player_index)  # Simula la mossa sullo stato

        next_state_rep = self.state_representation(next_state)
        next_max = max(self.q_table[next_state_rep].values(), default=0) if next_state_rep in self.q_table else 0

        self.update_q_table(state, move, reward, next_max)

    def evaluate_move_effect(self, game, move):
        current_board = game.get_board()
        ql_max_elements_before, ql_otk_before = count_aligned(current_board, self.player_index)
        opp_max_elements_before, opp_otk_before = count_aligned(current_board, 1 - self.player_index)

        # Simula la mossa
        new_board = apply_move(deepcopy(current_board), move, self.player_index)
        ql_max_elements_after, ql_otk_after = count_aligned(new_board, self.player_index)
        opp_max_elements_after, opp_otk_after = count_aligned(new_board, 1 - self.player_index)

        # reward basato su differenza e prevenzione di mosse critiche
        reward = 0

        # Miglioramento della posizione
        if ql_max_elements_after > ql_max_elements_before:
            reward += (ql_max_elements_after - ql_max_elements_before) * 2

        # Blocco dell'avversario
        if opp_otk_before and not opp_otk_after:
            reward += 2

        # Vittoria imminente
        if ql_otk_after and not ql_otk_before:
            reward += 3

        # Vittoria
        if ql_max_elements_after == 5:
            reward += 6

        # Avversario vicino alla vittoria
        if not opp_otk_before and opp_otk_after:
            reward -= 2

        # Sconfitta
        if opp_max_elements_after > 4:
            reward -= 5

        # mosse neutre
        if ql_max_elements_after <= ql_max_elements_before and not ql_otk_after:
            reward -= 1

        return reward

    
    def finalize_game(self, result):
        if result == "win":
            self.win_count += 1
            reward = 10  # vittoria
        elif result == "draw":
            reward = 1  # pareggio
            self.draw_count += 1
        else:  # "lose"
            reward = -10  # sconfitta
            self.loss_count += 1

        if (self.win_count + self.draw_count + self.loss_count) % 10 == 0:
            self.win_count = 0
            self.draw_count = 0
            self.loss_count = 0

        # Aggiorna la Q-table retroattivamente sulle mosse effettuate durante la partita
        for state, action, next_state in reversed(self.moves_history):
            current_q_value = self.q_table.get(state, {}).get(self.move_to_key(action), 0)
            next_max = 0
            if next_state in self.q_table:
                next_max = max(self.q_table[next_state].values())
            self.q_table.setdefault(state, {})[self.move_to_key(action)] = current_q_value + self.alpha * (reward + self.gamma * next_max - current_q_value)

        # Resetta la history delle mosse per la prossima partita
        self.moves_history.clear()

        
def train_qlearning_player(player, opponent, num_games=10000, save_interval=100):
    wins, last_wins, draws, last_draws, losses, last_losses = 0, 0, 0, 0, 0, 0

    for game_number in tqdm(range(num_games)):
        game = Game() 
        winner = game.play(player, opponent)
        
        if winner == player.player_index:
            wins += 1
            last_wins += 1
            game_result = "win"
        elif winner == -2: 
            draws += 1
            last_draws += 1
            game_result = "draw"
        else:
            losses += 1
            last_losses += 1
            game_result = "lose"
        player.finalize_game(game_result)
        #print(f"Vittorie: {wins}, Pareggi: {draws}, Sconfitte: {losses}")

        
        # Salva periodicamente la Q-table
        if (game_number + 1) % save_interval == 0:
            print(f"Salvataggio progressi dopo {game_number + 1} partite...")
            print(f"Vittorie: {100*last_wins/save_interval}, Pareggi: {100*last_draws/save_interval}, Sconfitte: {100*last_losses/save_interval}")
            last_wins, last_draws, last_losses = 0, 0, 0
            player.save_q_table()

    print(f"Training completato dopo {num_games} partite")
    print(f"Vittorie: {wins}, Pareggi: {draws}, Sconfitte: {losses}")
    #print(f"Media mosse per pareggio: {total_moves_in_draws / draws}")
    player.save_q_table()
        

def test_qlearning_player(player, opponent, num_games=100):
    wins, draws, losses, total_moves_in_draws = 0, 0, 0, 0

    for _ in tqdm(range(num_games)):
        game = Game() 
        winner = game.play(player, opponent)
        
        if winner == player.player_index:
            wins += 1
        elif winner == -2: 
            draws += 1
        else:
            losses += 1

    print(f"Test completato dopo {num_games} partite")
    print(f"Vittorie: {wins} ({wins/num_games*100:.2f}%)")
    print(f"Pareggi: {draws} ({draws/num_games*100:.2f}%)")
    print(f"Sconfitte: {losses} ({losses/num_games*100:.2f}%)")

if __name__ == "__main__":
    player = QLearningPlayer(player_index=0, preload=True)
    opponent = MiniMaxPlayer(1, 1)
    #opponent = RandomPlayer()
    training = False
    training = True

    if training:
        train_qlearning_player(player, opponent, num_games=400000, save_interval=100)
    else:
        test_qlearning_player(player, opponent, num_games=100)
