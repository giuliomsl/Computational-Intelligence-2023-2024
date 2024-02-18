from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from tqdm import tqdm
from qlearning import QLearningPlayer
from minimax_strategy import MiniMaxPlayer
from utils import *
from game import Game, Move, Player

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game) -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move
 
class DQNAgent(Player):
    def __init__(self, player_index, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)    # Buffer di riproduzione per memorizzare esperienze passate
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 1
        self.learning_rate = 0.001  # Tasso di apprendimento per l'ottimizzatore
        self.player_index = player_index
        self.model = self._build_model()
    
    def _build_model(self):
        # Costruisce il modello della rete neurale con struttura lineare
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        return model
    
    def save_model(self):
        torch.save(self.model.state_dict(), "dql_model.pth")
    
    def save_progress(self):
        progress = {
            "epsilon": self.epsilon,
            "memory": self.memory,
            # Aggiungi altri attributi se necessario
        }
        with open("dql_progress.pickle", 'wb') as f:
            pickle.dump(progress, f)

    def load_model(self):
        self.model.load_state_dict(torch.load("dql_model.pth"))
        self.model.eval()  # Imposta il modello in modalità di valutazione
    
    def load_progress(self):
        with open("dql_progress.pickle", 'rb') as f:
            progress = pickle.load(f)
            self.epsilon = progress["epsilon"]
            self.memory = progress["memory"]
    
    def remember(self, state, action, reward, next_state, done):
        # Memorizza l'esperienza passata nel buffer di riproduzione
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return np.argmax(act_values.detach().numpy()[0])  # returns action
    
    def replay(self, batch_size, gamma):
        # Effettua l'aggiornamento del modello utilizzando il buffer di riproduzione
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            # Convert state and next_state to PyTorch tensors
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Adds a batch dimension
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)  # Adds a batch dimension

            target = reward
            if not done:
                # Use next_state_tensor instead of next_state
                target = (reward + gamma * torch.max(self.model(next_state_tensor)).item())
            
            # Use state_tensor instead of state
            target_f = self.model(state_tensor)
            target_f[0][action] = target
            
            self.optimizer.zero_grad()
            # Calculate loss using the tensor, not the NumPy array
            loss = nn.MSELoss()(target_f, self.model(state_tensor))
            loss.backward()
            self.optimizer.step()


    def index_to_move(self, action_index, game):
        """
        Converte un indice di azione in una mossa specifica, assicurandosi che la mossa sia valida.
        """
        # Ottieni l'elenco attuale delle mosse possibili e valide per il giocatore corrente.
        possible_moves = get_possible_moves(game.get_board(), self.player_index)

        # Assicurati che l'indice dell'azione sia entro i limiti dell'elenco delle mosse possibili.
        if action_index >= 0 and action_index < len(possible_moves):
            return possible_moves[action_index]
        else:
            # Se l'indice di azione non è valido, scegli una mossa casuale dall'elenco delle mosse valide.
            # Questo caso può verificarsi se, per qualche motivo, l'azione scelta è fuori range.
            # È importante gestire questo caso per evitare errori di runtime.
            return random.choice(possible_moves)
        
    def calculate_reward(self, game):
        winner = game.check_winner()
        if winner == self.player_index:
            # Ricompensa positiva per la vittoria
            return 1
        elif winner == -2:
            # Ricompensa neutra o leggermente negativa per il pareggio
            return 0
        elif winner != -1:
            # Penalità per la sconfitta
            return -1
        else:
            # Nessuna ricompensa o penalità immediata se il gioco non è ancora finito
            # Potresti voler aggiungere logica qui per ricompense intermedie basate sullo stato del gioco
            return 0

    def step(board, action, player_index):
        
        # Esegue un passo nel gioco data un'azione e restituisce il nuovo stato, la ricompensa e se c'è un vincitore
        
        new_board = apply_move(deepcopy(board), action, player_index)
        winner = check_winner(new_board)
        
        if winner == player_index:
            reward = 2  
            done = True
        elif winner == -2:
            reward = 0 
            done = True
        elif winner != -1:
            reward = -1  
            done = True
        else:
            reward = 0 
            done = False
        
        return new_board, reward, done

    
    def make_move(self, game):
        state = normalize_state_simple(game.get_board(), game.get_current_player())
        action_index = self.act(state)
        chosen_move = self.index_to_move(action_index, game)
        (x, y), direction = chosen_move
        swapped_position = (y, x)
        swapped_move = (swapped_position, direction)
        return swapped_move

    
    def train_dql_player(self, episodes, opponent, batch_size=32, gamma=0.95, epsilon_decay=0.995, epsilon_min=0.01):
        wins, last_wins, draws, last_draws, losses, last_losses = 0, 0, 0, 0, 0, 0  # Initialize counters for game outcomes
        
        for episode in tqdm(range(1, episodes +1)):
            game = Game() 
            current_player_idx = 0 
            done = False
            
            while not done:
                if current_player_idx == self.player_index:
                    state = normalize_state_simple(game.get_board(), self.player_index)
                    action = self.act(state)
                    move = self.index_to_move(action, game)
                    game.play(self, opponent)
                    
                    new_state = normalize_state_simple(game.get_board(), self.player_index)
                    reward = self.calculate_reward(game)
                    done = game.check_winner() != -1  
                    self.remember(state, action, reward, new_state, done)
                else:
                    game.play(opponent, self) 
                
                current_player_idx = 1 - current_player_idx 

                if len(self.memory) > batch_size:
                    self.replay(batch_size, gamma) 

                outcome = game.check_winner()
                if outcome == self.player_index:
                    wins += 1
                    last_wins += 1
                elif outcome == -2:  
                    draws += 1
                    last_draws += 1
                else:
                    losses += 1
                    last_losses += 1

            if episode % 500 == 0:  
                last_total_games = last_wins + last_draws + last_losses
                print(f"\nEpisode: {episode}")
                print(f"Wins: {last_wins / last_total_games * 100:.2f}%, Draws: {last_draws / last_total_games * 100:.2f}%, Losses: {last_losses / last_total_games * 100:.2f}%")
                self.save_model()
                self.save_progress()
                last_wins, last_draws, last_losses = 0, 0, 0

            self.epsilon = max(self.epsilon * epsilon_decay, epsilon_min) 
        
        print(f"After {episodes} episodes: Wins: {wins}, Draws: {draws}, Losses: {losses}")

    def test_dql_player(self, opponent, num_games=100):
        wins, draws, losses = 0, 0, 0
        for _ in tqdm(range(num_games)):
            game = Game()  
            winner = game.play(self, opponent)
            
            if winner == self.player_index:
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
    player = DQNAgent(player_index=0, state_size=25, action_size=100)
    #opponent = RandomPlayer()
    opponent = QLearningPlayer(player_index=1, preload=True)
    training = False
    #training = True
    player.load_model()
    if training:
        episodes = 10000
        player.train_dql_player(episodes, opponent, batch_size=32, gamma=0.95, epsilon_decay=1, epsilon_min=0.01)
        #player.save_model()
    else:
        player.test_dql_player(opponent)

'''
Testing versus RandomPlayer
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 168.05it/s]
Test completato dopo 100 partite
Vittorie: 70 (70.00%)
Pareggi: 0 (0.00%)
Sconfitte: 30 (30.00%)


Training versus QLearningPlayer
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 53.74it/s]
Test completato dopo 100 partite
Vittorie: 74 (74.00%)
Pareggi: 0 (0.00%)
Sconfitte: 26 (26.00%)
'''