# Lab10.ipynb

## Tic-Tac-Toe Q-Learning Project

Implementation of a Tic-Tac-Toe game where a Q-Learning player (`QLearningPlayer`) is trained to play against a random opponent (`RandomPlayer`).

## Project Structure

- `TicTacToe`: A class representing the Tic-Tac-Toe game environment.
- `RandomPlayer`: An agent that makes moves by randomly choosing from available spots on the board.
- `QLearningPlayer`: A rl agent that uses Q-Learning to learn the game strategy by playing against the `RandomPlayer`.

## Training

The `QLearningPlayer` is trained using a training loop where it plays against the `RandomPlayer` across a specified number of episodes.

Training involves:
- Alternating the starting player between `QLearningPlayer` and `RandomPlayer` to ensure balanced learning.
- Saving the trained Q-Table to a `.pkl` file for later use and evaluation.

## Playing Against the Q-Learning Agent

After training, you can play against the trained `QLearningPlayer` by running the `Me_vs_opponent` function. Just input your moves as numbers from 1 to 9, corresponding to positions on the Tic-Tac-Toe board, and compete directly against the AI.

## Results

```
Testing win rate: 100%|██████████| 1000/1000 [00:01<00:00, 756.95it/s]
Win rate: 77.60%
Total games: 1000, Wins: 776, Losses: 165, Draws: 59
```
