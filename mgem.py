import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm
import logging
import copy
import math
from datetime import datetime

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Point-2 Game Logic
# =========================

class TicTacToeBoard:
    PLAYER_MAP = {1: "X", -1: "O"}
    STATE_MAP = {1: "unfinished", 2: "draw", 3: "finished"}
    SECTION_MAP = {
        1: [0, 0],
        2: [0, 3],
        3: [0, 6],
        4: [3, 0],
        5: [3, 3],
        6: [3, 6],
        7: [6, 0],
        8: [6, 3],
        9: [6, 6],
        10: [0, 0]
    }
    SECTION_WON_REWARD = 20  # Reward for winning a section
    SECTION_DRAW_REWARD = 5  # Reward for drawing a section
    
    def __init__(self):
        self.reset_board()
        
    def reset_board(self):
        """Reset all game boards to their initial state"""
        self.grid = np.zeros((9, 9), dtype=np.int8)
        self.section_grid = np.zeros((3, 3), dtype=np.int8)
        self.section_state_grid = np.ones((3, 3), dtype=bool)
        self.possible_moves_grid = np.ones((9, 9), dtype=bool)

        self.possible_moves = np.ones((9, 9), dtype=bool)

        self.grid_filter = np.ones((9, 9), dtype=bool)
        self.current_player = 1
        self.current_reward = 0
    
    def print_board(self, grid: np.ndarray = None):
        """Print the game board with symbols"""
        if grid is None:
            grid = self.grid
            
        symbols = {0: '·', 1: 'x', -1: 'o'}
        for row in grid:
            print(" ".join([symbols[x] for x in row]))
    
    def print_possible_moves_board(self, grid: np.ndarray = None):
        """Print the possible moves board with symbols"""
        if grid is None:
            grid = self.possible_moves_grid
            
        symbols = {False: '❌', True: '❎'}
        for row in grid:
            print(" ".join([symbols[x] for x in row]))
    
    @staticmethod
    def get_row_col(index: int) -> tuple[int, int]:
        """Convert linear index to row and column"""
        row = index // 9
        col = index % 9
        return (row, col)
    
    @staticmethod
    def get_section(row: int, col: int) -> int:
        """Get section number from row and column"""
        return (row // 3) * 3 + (col // 3) + 1
    
    @staticmethod
    def get_local_index(row: int, col: int) -> int:
        """Get local index within a section"""
        local_row = row % 3
        local_col = col % 3
        return local_row * 3 + local_col + 1
    
    def map_section(self, section_number: int) -> list[int]:
        """Get starting indices for a section"""
        return self.SECTION_MAP.get(section_number, "Invalid section number")
    
    def create_bool_grid_filter(self, section_number: int) -> np.ndarray:
        """Create a filter for possible moves in a specific section"""
        grid_filter = np.zeros((9, 9), dtype=bool)
        section_indices = self.map_section(section_number)
        if isinstance(section_indices, str):
            return grid_filter
        row_start = section_indices[0]
        col_start = section_indices[1]
        grid_filter[row_start:row_start+3, col_start:col_start+3] = \
            self.possible_moves_grid[row_start:row_start+3, col_start:col_start+3]
        return grid_filter
    
    def move(self, player: int, index: int):
        """Make a move on the board"""
        row, col = self.get_row_col(index)
        self.grid[row, col] = player
        self.possible_moves_grid[row, col] = False
    
    @staticmethod
    def get_diagonal_sums(section: np.ndarray) -> tuple[int, int]:
        """Calculate sums of both diagonals in a section"""
        main_diag_sum = section[0, 0] + section[1, 1] + section[2, 2]
        anti_diag_sum = section[0, 2] + section[1, 1] + section[2, 0]
        return (main_diag_sum, anti_diag_sum)
    
    def extract_section(self, grid: np.ndarray, section_number: int) -> np.ndarray:
        """Extract a 3x3 section from the main grid"""
        section_indices = self.map_section(section_number)
        if isinstance(section_indices, str):
            return np.zeros((3, 3), dtype=np.int8)
        initial_row_index = section_indices[0]
        initial_col_index = section_indices[1]
        return grid[initial_row_index:initial_row_index+3, initial_col_index:initial_col_index+3]
    
    def check_section_state(self, section: np.ndarray) -> int:
        """Check the state of a section (unfinished, draw, or won)"""
        row_sum = np.abs(np.sum(section, axis=1))
        col_sum = np.abs(np.sum(section, axis=0))
        diagonal_sum = np.abs(self.get_diagonal_sums(section))
        
        if 3 in col_sum or 3 in row_sum or 3 in diagonal_sum:
            return 3  # Won
        elif 0 in section:
            return 1   # Unfinished
        else:
            return 2   # Draw
    
    def clear_section(self, grid: np.ndarray, section_number: int):
        """Clear a section in the given grid"""
        row_start = ((section_number - 1) // 3) * 3
        col_start = ((section_number - 1) % 3) * 3
        grid[row_start:row_start+3, col_start:col_start+3] = False
        self.grid[row_start:row_start+3, col_start:col_start+3] = self.current_player
    
    def update_section_grid(self, section_number: int, value: int):
        """Update the section grid with a player's win"""
        row_index = (section_number - 1) // 3
        col_index = (section_number - 1) % 3
        self.section_grid[row_index, col_index] = value
    
    def update_section_state_grid(self, section_number: int):
        """Mark a section as completed in the state grid"""
        row_index = (section_number - 1) // 3
        col_index = (section_number - 1) % 3
        self.section_state_grid[row_index, col_index] = False
    
    def print_possible_moves(self, is_whole_board_active: bool, section_number: int, print_output = True):
        """Print the current possible moves"""
        if print_output:
            print("General state:")
            self.print_possible_moves_board(self.section_state_grid)
            print("Specific state:")
        if is_whole_board_active:
            if print_output:
                self.print_possible_moves_board(self.possible_moves_grid)
            return self.possible_moves_grid
        else:
            filter_grid = self.create_bool_grid_filter(section_number)
            if print_output:
                self.print_possible_moves_board(filter_grid)
            return filter_grid
        
    
    def check_if_move_possible(self, is_whole_board_active: bool, index: int, section_number: int) -> bool:
        """Check if a move is valid"""
        row, col = self.get_row_col(index)
        if is_whole_board_active:
            return self.possible_moves_grid[row, col]
        else:
            filter_grid = self.create_bool_grid_filter(section_number)
            return filter_grid[row, col]
    
    def post_move_transformations(self, index: int, player: int) -> tuple[bool, int]:
        """Update game state after a move"""
        row, col = self.get_row_col(index)
        section_number = self.get_section(row, col)
        local_index = self.get_local_index(row, col)
        section = self.extract_section(self.grid, section_number)
        section_state = self.check_section_state(section)

        if section_state == 3:  # Section won
            self.update_section_grid(section_number, player)
            self.update_section_state_grid(section_number)
            self.clear_section(self.possible_moves_grid, section_number)
            self.current_reward = self.SECTION_WON_REWARD * self.current_player
        elif section_state == 2:  # Section draw
            self.update_section_state_grid(section_number)
            self.clear_section(self.possible_moves_grid, section_number)
            self.current_reward = self.SECTION_DRAW_REWARD
        else:  # Section unfinished
            self.current_reward = -self.current_player

        # Determine next active section
        local_section_row_index = (local_index - 1) // 3
        local_section_col_index = (local_index - 1) % 3
        is_next_section_free = self.section_state_grid[local_section_row_index, local_section_col_index]
        self.current_player *= -1

        if not is_next_section_free:
            self.grid_filter = self.possible_moves_grid
            return True, 0  # Whole board active
        else:
            self.grid_filter = self.create_bool_grid_filter(local_index)
            return False, local_index  # Specific section active


class TicTacToeGame:
    def __init__(self):
        self.board = TicTacToeBoard()
        self.current_player = self.board.current_player
        self.is_whole_board_active = True
        self.active_section = 0
        self.last_move = None
        self.state = 0  # 0: ongoing, 1: player 1 won, -1: player 2 won, 2: draw
    
    def get_user_move(self) -> int:
        """Get valid move input from user"""
        while True:
            try:
                user_move_str = input("Enter your move (0-80): ")
                index = int(user_move_str)
                if 0 <= index <= 80:
                    return index
                print("Invalid move. Please enter a number between 0 and 80.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def make_move(self, index):
        if self.board.check_if_move_possible(self.is_whole_board_active, index, self.active_section):
            self.board.move(self.board.current_player, index)
            self.is_whole_board_active, self.active_section = \
                self.board.post_move_transformations(index, self.board.current_player)
            self.last_move = index
            return True
        else:
            return False

    def check_winner(self):
        # Check if the game is won (3 in a row in the section grid)
        section_grid = self.board.section_grid
        row_sum = np.abs(np.sum(section_grid, axis=1))
        col_sum = np.abs(np.sum(section_grid, axis=0))
        diag1 = abs(section_grid[0,0] + section_grid[1,1] + section_grid[2,2])
        diag2 = abs(section_grid[0,2] + section_grid[1,1] + section_grid[2,0])
        
        if 3 in row_sum or 3 in col_sum or diag1 == 3 or diag2 == 3:
            self.state = self.board.current_player * -1  # Winner is the player who just moved
            return self.state
        
        # Check if game is a draw (all sections are completed)
        if not np.any(self.board.section_state_grid):
            self.state = 2
            return "Tie"
        
        return 0
    
    def get_move_filter(self):
        return self.board.grid_filter.flatten().tolist()
    
    def play(self):
        """Main game loop (manual play)"""
        while True:
            self.board.print_possible_moves(self.is_whole_board_active, self.active_section)
            self.board.print_board()
            
            index = self.get_user_move()
            if not self.board.check_if_move_possible(self.is_whole_board_active, index, self.active_section):
                print(f"Move to [{index}] is impossible")
                continue

            self.board.move(self.current_player, index)
            self.is_whole_board_active, self.active_section = \
                self.board.post_move_transformations(index, self.current_player)

            game_section_state = self.board.check_section_state(self.board.section_grid)
            if game_section_state == 3:  # Game won
                self.board.print_board()
                print(f"Game finished, player {self.board.PLAYER_MAP[self.current_player]} won!")
                break
            elif game_section_state == 2:  # Game draw
                self.board.print_board()
                print("Game finished, tie!")
                break

            self.current_player *= -1  # Switch player


# =========================
# RL Environment (wraps point-2)
# =========================

class UltimateTicTacToeEnv:
    """RL environment wrapper for Ultimate Tic Tac Toe using point-2 logic."""
    def __init__(self):
        self.game = TicTacToeGame()
        self.action_space = 81  # 0-80
        self.observation_space = (9, 9)  # Board shape
        self.valid_actions = self.get_valid_actions()

    def reset(self):
        del self.game
        self.game = TicTacToeGame()
        obs = self._get_obs()
        return obs

    def step(self, action, force_legal_move=False):
        valid_actions = self.get_valid_actions()
        if action not in valid_actions:
            # Invalid moves are strongly penalized and terminate
            obs = self._get_obs()
            return obs, -100.0, True, {'winner': -self.game.board.current_player, 'valid': False}

        self.valid_actions = valid_actions
        self.game.make_move(action)
        winner = self.game.check_winner()
        obs = self._get_obs()
        if winner == 1:
            reward = 100.0     # point-2 terminal reward
            done = True
        elif winner == -1:
            reward = -100.0    # point-2 terminal reward
            done = True
        elif winner == "Tie":
            reward = 0.5       # point-2 terminal reward
            done = True
        else:
            reward = float(self.game.board.current_reward)  # point-2 per-move/section rewards
            done = False
        return obs, reward, done, {"winner" : winner, "valid": True}

    def render(self):
        self.game.board.print_board()

    def get_valid_actions(self):
        valid_moves = self.game.board.print_possible_moves(self.game.is_whole_board_active, self.game.active_section, False)
        flat_mask = valid_moves.flatten()
        valid_indices = np.where(flat_mask)[0]
        return valid_indices.tolist()

    def _get_obs(self):
        # Return a copy of the board state as observation
        return (np.copy(self.game.board.grid), np.copy(self.game.board.section_grid), np.copy(self.game.board.grid_filter))


# =========================
# Helper: state -> tensor (kept from your original)
# =========================

def convert_state_to_tensor(state, player):
    """
    Converts a state tuple into a PyTorch tensor with the correct
    channel-first format for the network.

    Args:
        state (tuple): (main_board, section_board, valid_mask)
        player (int): 1 or -1, the current player's turn.

    Returns:
        torch.Tensor: A tensor of shape (1, 3, 9, 9).
    """
    board, macroboard, valid_mask = state

    # Flip perspective so the current player is always '1'
    p_board = board * player
    p_macroboard = macroboard * player

    # Expand 3x3 macro to 9x9
    expanded_macro = np.repeat(p_macroboard, 3, axis=0).repeat(3, axis=1)

    stacked_state = np.stack(
        [p_board, expanded_macro, valid_mask.astype(np.float32)],
        axis=0
    )
    tensor_state = torch.from_numpy(stacked_state).float().unsqueeze(0)
    return tensor_state


# =========================
# Replay Buffer
# =========================

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# =========================
# DQN Network
# =========================

class DQNetwork(nn.Module):
    """Deep Q-Network for Ultimate Tic-Tac-Toe."""
    def __init__(self, input_shape=(3, 9, 9), n_actions=81):
        super(DQNetwork, self).__init__()
        c, h, w = input_shape
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(c, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * h * w, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# =========================
# MCTS (wired to env.get_valid_actions + network eval)
# =========================

class MCTSNode:
    """A node in the Monte Carlo Tree."""
    def __init__(self, env_snapshot, parent=None, action=None):
        self.env = env_snapshot          # a deep-copied env
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_value = 0.0
        self.untried_actions = self.env.get_valid_actions()
        self.is_terminal = False

    def ucb1(self, c_param=1.4):
        if self.visits == 0:
            return float('inf')
        return (self.total_value / self.visits) + c_param * math.sqrt(math.log(self.parent.visits + 1) / self.visits)


class MCTS:
    def __init__(self, agent_network, c_param=1.4):
        self.net = agent_network
        self.c_param = c_param
        self.device = agent_network.device

    def search(self, root_env, num_simulations=100):
        root = MCTSNode(copy.deepcopy(root_env), parent=None, action=None)

        # If no moves, return anything
        if not root.untried_actions:
            return random.randint(0, 80)

        for _ in range(num_simulations):
            node = self._select(root)
            if not node.is_terminal:
                node = self._expand(node)
            value = self._evaluate(node)
            self._backpropagate(node, value)

        # Choose by visit count
        best_child = max(root.children, key=lambda c: c.visits) if root.children else None
        return best_child.action if best_child else random.choice(root.untried_actions)

    def _select(self, node):
        # Traverse down by UCB1 until we hit a node we can expand or terminal
        while True:
            if node.is_terminal or node.untried_actions:
                return node
            if not node.children:
                return node
            node = max(node.children, key=lambda ch: ch.ucb1(self.c_param))

    def _expand(self, node):
        if not node.untried_actions:
            return node
        action = node.untried_actions.pop()
        next_env = copy.deepcopy(node.env)
        _, _, done, _ = next_env.step(action)
        child = MCTSNode(next_env, parent=node, action=action)
        child.is_terminal = done
        node.children.append(child)
        return child

    def _evaluate(self, node):
        # If terminal, value is immediate outcome from current env perspective
        # Map env rewards to a value: we'll do a single-network value estimate for non-terminal
        if node.is_terminal:
            # Use a simple terminal scoring consistent with env: winner already baked into env state.
            # We'll re-evaluate with network to keep scale consistent.
            pass

        # Network Q-max over valid moves as a proxy value
        with torch.no_grad():
            state = node.env._get_obs()
            player = node.env.game.board.current_player
            state_tensor = convert_state_to_tensor(state, player).to(self.device)
            q_values = self.net(state_tensor).squeeze(0).cpu().numpy()

            valid = np.full(81, False)
            valid[node.env.get_valid_actions()] = True
            q_values[~valid] = -1e9
            value = float(np.max(q_values))
        return value

    def _backpropagate(self, node, value):
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.total_value += value
            value = -value  # alternate perspective each ply
            cur = cur.parent


# =========================
# DQN Agent (keeps your structure)
# =========================

class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.n_actions = self.env.action_space
        self.gamma = 0.99
        self.epsilon = 1.0   # used minimally since we rely on MCTS
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.learning_rate = 1e-4
        self.target_update_frequency = 1000
        
        self.q_network = DQNetwork()
        self.target_network = DQNetwork()
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.steps_done = 0
        self.mcts = MCTS(self.q_network, c_param=1.4)

    def get_action(self, state, player, valid_actions, num_simulations=100):
        # Use MCTS, which uses env snapshots and valid moves
        return self.mcts.search(self.env, num_simulations)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        experiences = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        # NOTE: Keeping your original structure:
        # states viewed as player=1, next_states as player=-1
        states_t = torch.cat([convert_state_to_tensor(s, 1) for s in states]).to(self.q_network.device)
        actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(-1).to(self.q_network.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.q_network.device)
        next_states_t = torch.cat([convert_state_to_tensor(s, -1) for s in next_states]).to(self.q_network.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(self.q_network.device)
        
        q_values = self.q_network(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states_t).max(1)[0]
            target_q_values = rewards_t + self.gamma * next_q_values * (1 - dones_t)
            
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps_done += 1
        
        if self.steps_done % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# =========================
# Training Loop
# =========================

# def train(agent, env, num_episodes=10000):
#     logger.info("Starting training...")
#     for episode in tqdm(range(num_episodes)):
#         state = env.reset()
#         done = False
#         total_reward = 0.0
        
#         while not done:
#             # Player 1's turn
#             valid_moves_p1 = env.get_valid_actions()
#             action_p1 = agent.get_action(state, env.game.board.current_player, valid_moves_p1, num_simulations=200)
#             next_state_p1, reward_p1, done, info_p1 = env.step(action_p1)
            
#             agent.replay_buffer.push(state, action_p1, reward_p1, next_state_p1, done)
#             agent.learn()
            
#             state = next_state_p1
#             total_reward += reward_p1
#             if done:
#                 break
                
#             # Player -1's turn (self-play)
#             valid_moves_p_1 = env.get_valid_actions()
#             action_p_1 = agent.get_action(state, env.game.board.current_player, valid_moves_p_1, num_simulations=200)
#             next_state_p_1, reward_p_1, done, info_p_1 = env.step(action_p_1)
            
#             # Zero-sum perspective (keep structure; you used negative here)
#             agent.replay_buffer.push(state, action_p_1, -reward_p_1, next_state_p_1, done)
#             agent.learn()
            
#             state = next_state_p_1
#             total_reward += reward_p_1
#             if done:
#                 break
        
#         agent.update_epsilon()

#         # Save periodically
#         if (episode + 1) % 500 == 0:
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             filename = f"ultimate_ttt_agent_episode_{episode+1}_{timestamp}.pt"
#             torch.save(agent.q_network.state_dict(), filename)
#             logger.info(f"Model saved as {filename} (episode {episode+1}, total_reward={total_reward:.2f})")

#     logger.info("Training complete.")
#     torch.save(agent.q_network.state_dict(), "ultimate_ttt_agent_final.pt")
#     logger.info("Final model saved as ultimate_ttt_agent_final.pt")

import os
from datetime import datetime

def train(agent, env, num_episodes=10000):
    logger.info("Starting training...")
    
    # Ensure save directory exists
    SAVE_DIR = "saved_models"
    os.makedirs(SAVE_DIR, exist_ok=True)

    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            # Player 1's turn
            valid_moves_p1 = env.get_valid_actions()
            action_p1 = agent.get_action(state, env.game.board.current_player, valid_moves_p1, num_simulations=200)
            next_state_p1, reward_p1, done, info_p1 = env.step(action_p1)
            
            agent.replay_buffer.push(state, action_p1, reward_p1, next_state_p1, done)
            agent.learn()
            
            state = next_state_p1
            total_reward += reward_p1
            if done:
                break
                
            # Player -1's turn (self-play)
            valid_moves_p_1 = env.get_valid_actions()
            action_p_1 = agent.get_action(state, env.game.board.current_player, valid_moves_p_1, num_simulations=200)
            next_state_p_1, reward_p_1, done, info_p_1 = env.step(action_p_1)
            
            # Zero-sum perspective
            agent.replay_buffer.push(state, action_p_1, -reward_p_1, next_state_p_1, done)
            agent.learn()
            
            state = next_state_p_1
            total_reward += reward_p_1
            if done:
                break
        
        agent.update_epsilon()

        # Save every 500 episodes
        if (episode + 1) % 500 == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ultimate_ttt_agent_ep{episode+1}_{timestamp}.pt"
            save_path = os.path.join(SAVE_DIR, filename)
            torch.save(agent.q_network.state_dict(), save_path)
            logger.info(f"Model saved at {save_path} (episode {episode+1}, total_reward={total_reward:.2f})")

    logger.info("Training complete.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ultimate_ttt_agent_ep{episode+1}_{timestamp}.pt"
    final_model_path = os.path.join(SAVE_DIR, filename)
    torch.save(agent.q_network.state_dict(), final_model_path)
    logger.info(f"Final model saved at {final_model_path}")


if __name__ == "__main__":
    env = UltimateTicTacToeEnv()
    agent = DQNAgent(env)

    # Train (adjust episodes/simulations for speed while testing)
    train(agent, env, num_episodes=30_000)