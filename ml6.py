import numpy as np
import random

# Initialize the Q-table
# Each state (board configuration) stores 9 Q-values, one for each position
Q = {}

# Hyperparameters
alpha = 0.5       # Learning rate: controls how much new information updates old Q-values
gamma = 0.9       # Discount factor: importance given to future rewards
epsilon = 1.0     # Exploration probability: chance of selecting a random action

# Check whether a player has won or the game is a draw
def check_winner(board):
    win_patterns = [
        [0,1,2], [3,4,5], [6,7,8],
        [0,3,6], [1,4,7], [2,5,8],
        [0,4,8], [2,4,6]
    ]
    for pattern in win_patterns:
        a, b, c = pattern
        if board[a] == board[b] == board[c] != ' ':
            return board[a]
    if ' ' not in board:
        return 'Draw'
    return None

# Display the Tic-Tac-Toe board
def display_board(board):
    print()
    for i in range(0, 9, 3):
        print(" | ".join(board[i:i+3]))
        if i < 6:
            print("---------")
    print()

# Get Q-values for a state; initialize with zeros if state is new
def get_q_values(state):
    if state not in Q:
        Q[state] = np.zeros(9)
    return Q[state]

# Select an action using the epsilon-greedy policy
def choose_action(state, board, epsilon):
    available = [i for i in range(9) if board[i] == ' ']
    # Exploration: randomly select a move
    if random.random() < epsilon:
        return random.choice(available)
    # Exploitation: select the move with the highest Q-value
    q_values = get_q_values(state)
    best_value = max(q_values[i] for i in available)
    best_actions = [i for i in available if q_values[i] == best_value]
    return random.choice(best_actions)

# Train the Q-learning agent
episodes = 50000
for episode in range(episodes):
    board = [' '] * 9
    done = False
    while not done:
        # Agent plays as X
        state = ''.join(board)
        action = choose_action(state, board, epsilon)
        board[action] = 'X'
        result = check_winner(board)

        # Assign reward based on the game result
        if result == 'X':
            reward = 1       # Positive reward for winning
            done = True
        elif result == 'Draw':
            reward = 0.5     # Small positive reward for a draw
            done = True
        else:
            # Opponent plays as O using random available moves
            available = [i for i in range(9) if board[i] == ' ']
            if available:
                opponent_action = random.choice(available)
                board[opponent_action] = 'O'
            result = check_winner(board)

            if result == 'O':
                reward = -1  # Negative reward for losing
                done = True
            elif result == 'Draw':
                reward = 0.5
                done = True
            else:
                reward = 0

        # Observe the next state after the agent and opponent move
        next_state = ''.join(board)
        old_value = get_q_values(state)[action]

        # Calculate the maximum future Q-value
        if done:
            future_value = 0  # No future state after the game ends
        else:
            available = [i for i in range(9) if board[i] == ' ']
            future_value = max(get_q_values(next_state)[i] for i in available)

        # Q-learning update formula
        # Q(s,a) = Q(s,a) + alpha * [reward + gamma * max(Q(next_state)) - Q(s,a)]
        new_value = old_value + alpha * (
            reward + gamma * future_value - old_value
        )
        Q[state][action] = new_value

    # Decrease exploration gradually so the agent uses learned knowledge more often
    epsilon = max(0.05, epsilon * 0.9999)

    # Decrease learning rate gradually during training
    # Early training learns quickly; later training makes smaller adjustments
    alpha = max(0.05, 0.5 * (1 - episode / episodes))

# Test the trained agent against a random opponent
wins = 0
losses = 0
draws = 0
test_games = 1000

for game in range(test_games):
    board = [' '] * 9
    while True:
        # Agent uses its learned Q-values without exploration
        state = ''.join(board)
        action = choose_action(state, board, 0)
        board[action] = 'X'
        result = check_winner(board)

        if result == 'X':
            wins += 1
            break
        if result == 'Draw':
            draws += 1
            break

        # Random opponent plays as O
        available = [i for i in range(9) if board[i] == ' ']
        opponent_action = random.choice(available)
        board[opponent_action] = 'O'
        result = check_winner(board)

        if result == 'O':
            losses += 1
            break
        if result == 'Draw':
            draws += 1
            break

# Display testing results
print("Testing Results")
print("-------------------")
print("Wins:", wins)
print("Losses:", losses)
print("Draws:", draws)
print("Win Rate:", wins / test_games * 100, "%")
