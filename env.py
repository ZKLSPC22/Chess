import gymnasium as gym
import chess
import torch
import numpy as np
from gymnasium import spaces


class ChessEnv(gym.Env):
    _WHITE_PIECE_COLOR = '\033[1;34m'  # Bright blue for white pieces
    _BLACK_PIECE_COLOR = '\033[1;31m'  # Bright red for black pieces
    _LIGHT_SQUARE_BG = '\033[48;5;255m'  # Very light gray/white background
    _DARK_SQUARE_BG = '\033[48;5;240m'  # Medium gray background
    _RESET_COLOR = '\033[0m'

    def __init__(self):
        """Initializes the ChessEnv environment."""
        self.board = chess.Board()
        self.action_space = spaces.Discrete(4672)
        self.observation_space = spaces.Box(low=0, high=1, shape=(17, 8, 8), dtype=np.float32)

        self.piece_type_to_idx = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        self.idx_to_piece_type = {v: k for k, v in self.piece_type_to_idx.items()}
    
    def initial_state(self):
        return self._board_to_state(chess.Board())

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[torch.Tensor, dict]:
        """
        Resets the environment to its initial state.

        Args:
            seed (int | None): The seed for the random number generator. Defaults to None.
            options (dict | None): Additional options for resetting the environment. Defaults to None.

        Returns:
            tuple[torch.Tensor, dict]: A tuple containing the initial observation and an info dictionary.
        """
        super().reset(seed=seed)
        self.board.reset()
        obs = self._board_to_state(self.board)
        return obs, {}

    def step(self, state: torch.Tensor, action: int) -> tuple[torch.Tensor, float, bool, bool, dict]:
        if state.shape != (17, 8, 8):
            raise ValueError("State must be a 17x8x8 tensor")
        
        if action not in range(0, 4672):
            raise ValueError("Action must be an integer between 0 and 4672")

        board = self._state_to_board(state)
        from_square, to_square, promotion = self._decode_action_index(action)
        move = chess.Move(from_square, to_square, promotion=promotion)
        new_board = board.copy()
        if move in new_board.legal_moves:
            new_board.push(move)
            # Update the internal board object for rendering
            self.board = new_board
        else:
            return state, -10.0, False, False, {"error": "Illegal move"}
        
        terminated = new_board.is_game_over()
        reward = self._get_reward(new_board)
        new_state = self._board_to_state(new_board)
        return new_state, reward, terminated, False, {}

    def render(self, human_color: chess.Color):
        # Unicode characters for chess pieces
        unicode_pieces = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',
            None: ' ' 
        }
        rank_range = range(7, -1, -1) if human_color == chess.WHITE else range(8)
        file_range = range(8) if human_color == chess.WHITE else range(7, -1, -1)
        file_names = chess.FILE_NAMES if human_color == chess.WHITE else list(reversed(chess.FILE_NAMES))
        rank_names = chess.RANK_NAMES if human_color == chess.WHITE else list(reversed(chess.RANK_NAMES))
        
        # Print the board with files (columns) and ranks (rows)
        print("\n     " + "   ".join(file_names)) # Column letters with proper spacing
        print("   +" + "---" * 8 + "+") # Top border
        for rank_idx in rank_range: # Ranks 8 down to 1
            print(f" {rank_names[rank_idx]} |", end="")
            for file_idx in file_range: # Files A to H
                square = chess.square(file_idx, rank_idx)
                piece = self.board.piece_at(square)
                
                # Set square background color
                is_light_square = (rank_idx + file_idx) % 2 != 0
                bg_color = self._LIGHT_SQUARE_BG if is_light_square else self._DARK_SQUARE_BG
                
                # Get piece symbol and color
                piece_symbol = unicode_pieces[piece.symbol()] if piece else unicode_pieces[None]
                piece_color_code = ""
                if piece:
                    piece_color_code = self._WHITE_PIECE_COLOR if piece.color == chess.WHITE else self._BLACK_PIECE_COLOR
                
                print(f"{bg_color}{piece_color_code} {piece_symbol} {self._RESET_COLOR}", end="")
            print(f"| {rank_names[rank_idx]}{self._RESET_COLOR}")
        print("   +" + "---" * 8 + "+") # Bottom border
        print("     " + "   ".join(file_names)) # Column letters again with proper spacing

        # Show last move if available
        if self.board.move_stack:
            last_move = self.board.peek().uci()
            print(f"Last move: {last_move}")
        
        print(self._RESET_COLOR, end='') # Ensure terminal color is reset

    def get_legal_actions(self, state):
        board = self._state_to_board(state)
        legal_moves = [move.uci() for move in board.legal_moves]
        legal_actions = [self.encode_uci_to_action_index(move) for move in legal_moves]
        return legal_actions
    
    def encode_uci_to_action_index(self, uci: str) -> int:
        move = chess.Move.from_uci(uci)
        from_square = move.from_square
        to_square = move.to_square
        promotion = move.promotion
        if promotion is None:
            return from_square * 73 + to_square
        else:
            file = chess.square_file(to_square)
            promo_type = self.piece_type_to_idx[promotion]
            return from_square * 73 + 56 + file * 4 + promo_type
    
    def _get_reward(self, board: chess.Board) -> torch.Tensor:
        """
        Calculates the reward based on the game's outcome.

        Args:
            board (chess.Board): The current board state.

        Returns:
            torch.Tensor: A tensor representing the reward: 1.0 for White's win, 
                          -1.0 for Black's win, and 0.0 for a draw or ongoing game.
        """
        if board.is_game_over():
            outcome = board.outcome()
            if outcome is None: # Should not happen if game is over, but for safety
                return torch.tensor(0.0, dtype=torch.float32)
            if outcome.winner == chess.WHITE:
                return torch.tensor(1.0, dtype=torch.float32)
            elif outcome.winner == chess.BLACK:
                return torch.tensor(-1.0, dtype=torch.float32)
            else: # Draw
                return torch.tensor(0.0, dtype=torch.float32)
        # No reward for ongoing games
        return torch.tensor(0.0, dtype=torch.float32)

    def _board_to_state(self, board: chess.Board) -> torch.Tensor:
        """
        Converts a `chess.Board` object to a 17x8x8 tensor representation.

        The state tensor is structured as follows:
        - Planes 0-5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King).
        - Planes 6-11: Black pieces (Pawn, Knight, Bishop, Rook, Queen, King).
        - Plane 12: White king-side castling rights.
        - Plane 13: White queen-side castling rights.
        - Plane 14: Black king-side castling rights.
        - Plane 15: Black queen-side castling rights.
        - Plane 16: Side to move (1.0 for White, 0.0 for Black).

        Args:
            board (chess.Board): The board to convert.

        Returns:
            torch.Tensor: The 17x8x8 state tensor.
        """
        state = torch.zeros(17, 8, 8, dtype=torch.float32)
        
        # Populate piece planes
        for square, piece in board.piece_map().items():
            row, col = divmod(square, 8)
            plane_idx = self.piece_type_to_idx[piece.piece_type]
            if piece.color == chess.BLACK:
                plane_idx += 6  # Offset for black pieces
            state[plane_idx, row, col] = 1.0
        
        # Populate castling rights planes
        if board.has_kingside_castling_rights(chess.WHITE): state[12, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE): state[13, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK): state[14, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK): state[15, :, :] = 1.0

        # Populate side-to-move plane
        if board.turn == chess.WHITE: state[16, :, :] = 1.0

        return state
    
    def _state_to_board(self, state: torch.Tensor) -> chess.Board:
        if state.shape != (17, 8, 8):
            raise ValueError("State must be a 17x8x8 tensor")
        
        # Start with an empty board
        board = chess.Board(fen=None)
        
        # Set pieces based on the state tensor
        for plane_idx in range(12):
            piece_type_idx = plane_idx % 6
            color = chess.WHITE if plane_idx < 6 else chess.BLACK
            piece_type = self.idx_to_piece_type[piece_type_idx]
            piece = chess.Piece(piece_type, color)

            for row in range(8):
                for col in range(8):
                    if state[plane_idx, row, col] == 1.0:
                        board.set_piece_at(chess.square(col, row), piece)
        
        # Reconstruct castling rights from FEN string components
        castling_fen = ""
        if state[12, :, :].any(): castling_fen += "K"
        if state[13, :, :].any(): castling_fen += "Q"
        if state[14, :, :].any(): castling_fen += "k"
        if state[15, :, :].any(): castling_fen += "q"
        
        # The full FEN is not needed, just the castling part
        board.set_castling_fen(castling_fen if castling_fen else '-')
        
        # Set the turn
        if state[16, :, :].any():
            board.turn = chess.WHITE
        else:
            board.turn = chess.BLACK
        
        return board

    # 4672 -> 73 * 64
    def _decode_action_index(self, action: int):
        from_square = action // 73
        to_promo = action % 73
        if to_promo < 56:
            to_square = to_promo
            promotion = None
        else:
            to_square = (to_promo - 56) // 4 + (from_square // 8 == 6) * 8
            promo_type = (to_promo - 56) % 4
            promotion = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT][promo_type]
        return from_square, to_square, promotion
