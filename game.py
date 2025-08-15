import numpy as np

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
    SECTION_WON_REWARD = 20 # Reward for winning a section
    SECTION_DRAW_REWARD = 5 # Reward for drawing a section
    
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
    
    # def get_user_move(self) -> int:
    #     """Get valid move input from user"""
    #     while True:
    #         try:
    #             user_move_str = input("Enter your move (0-80): ")
    #             index = int(user_move_str)
    #             if 0 <= index <= 80:
    #                 return index
    #             print("Invalid move. Please enter a number between 0 and 80.")
    #         except ValueError:
    #             print("Invalid input. Please enter a number.")
    
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
        
    # def check_winner(self):
    #     game_section_state = self.board.check_section_state(self.board.section_grid)
    #     if game_section_state == 3:  # Game won
    #             self.state = self.board.current_player
    #             return self.board.current_player*(-1)
    #     elif game_section_state == 2:  # Game draw
    #             self.state = 2
    #             return "Tie"
    #     return 0

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
        """Main game loop"""
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


class UltimateTicTacToeEnv:
    """RL environment wrapper for Ultimate Tic Tac Toe."""
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
        if action not in self.get_valid_actions():
            raise ValueError(f"Invalid action: {action}. Valid actions are: {self.get_valid_actions()}")
        self.valid_actions = self.get_valid_actions()
        self.game.make_move(action)
        winner = self.game.check_winner()
        obs = self._get_obs()
        if winner == 1:
            reward = 100
            done = True
        elif winner == -1:
            reward = -100
            done = True
        elif winner == "Tie":
            reward = 0.5
            done = True
        else:
            reward = self.game.board.current_reward
            done = False
        return obs, reward, done, {"winner" : winner}

    def render(self):
        self.game.board.print_board()

    def get_valid_actions(self):
        valid_moves = self.game.board.print_possible_moves(self.game.is_whole_board_active, self.game.active_section, False)
        flat_mask = valid_moves.flatten()
        valid_indices = np.where(flat_mask)[0]
        return valid_indices

    def _get_obs(self):
        # Return a copy of the board state as observation
        return (np.copy(self.game.board.grid), np.copy(self.game.board.section_grid), np.copy(self.game.board.grid_filter))


if __name__ == "__main__":
    game = TicTacToeGame()
    game.play()