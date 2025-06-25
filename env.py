import gymnasium as gym
import chess
import torch
import numpy as np
from gymnasium import spaces
import sys
import os
import logging

logger = logging.getLogger(__name__)


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

        # Move encoding helpers based on AlphaZero
        self.queen_move_directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        self.knight_move_deltas = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
        self.underpromotion_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
    
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
        logger.info("Chess environment reset.")
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
        from_square, to_square, promotion = self._decode_action_index(action, board)
        move = chess.Move(from_square, to_square, promotion=promotion)
        move_uci = move.uci()
        new_board = board.copy()
        if move in new_board.legal_moves:
            new_board.push(move)
            # Update the internal board object for rendering
            self.board = new_board
            logger.debug(f"Step taken with legal move: {move_uci}. New FEN: {new_board.fen()}")
        else:
            logger.warning(f"Illegal move {move_uci} attempted. Board state unchanged.")
            return state, -10.0, False, False, {"error": "Illegal move"}
        
        terminated = new_board.is_game_over()
        reward = self._get_reward(new_board)
        new_state = self._board_to_state(new_board)
        return new_state, reward, terminated, False, {}

    def render(self, human_color: chess.Color):
        logger.debug("Rendering board.")
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

        # Optionally disable color on Windows if not supported
        use_color = True
        if os.name == 'nt':
            # Try to enable ANSI escape codes on Windows 10+
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except Exception:
                use_color = False
        def colorize(s, code):
            return f"{code}{s}{self._RESET_COLOR}" if use_color else s

        # Print the board with files (columns) and ranks (rows)
        # Centered file letters with 4-space wide cells
        print("\n     " + "   ".join(file_names))
        print("   +" + "----" * 8 + "+")
        for rank_idx in rank_range:  # Ranks 8 down to 1
            # First line of the double-row square (empty for vertical centering)
            print(f"   |", end="")
            for file_idx in file_range:
                is_light_square = (rank_idx + file_idx) % 2 != 0
                bg_color = self._LIGHT_SQUARE_BG if is_light_square else self._DARK_SQUARE_BG
                cell = "    "
                print(f"{bg_color if use_color else ''}{cell}{self._RESET_COLOR if use_color else ''}", end="")
            print(f"|   ")

            # Second line: rank number and piece symbol centered
            print(f" {rank_names[rank_idx]} |", end="")
            for file_idx in file_range:
                square = chess.square(file_idx, rank_idx)
                piece = self.board.piece_at(square)
                is_light_square = (rank_idx + file_idx) % 2 != 0
                bg_color = self._LIGHT_SQUARE_BG if is_light_square else self._DARK_SQUARE_BG
                
                piece_symbol = unicode_pieces[piece.symbol()] if piece else ' '
                piece_color_code = ""
                if piece:
                    piece_color_code = self._WHITE_PIECE_COLOR if piece.color == chess.WHITE else self._BLACK_PIECE_COLOR
                
                # Center the piece symbol in a 4-char wide cell
                cell = f" {piece_symbol}  "
                if use_color and piece:
                    cell = f"{piece_color_code}{cell}{self._RESET_COLOR}"

                print(f"{bg_color if use_color else ''}{cell}{self._RESET_COLOR if use_color else ''}", end="")
            print(f"| {rank_names[rank_idx]}")
        print("   +" + "----" * 8 + "+")  # Bottom border
        print("     " + "   ".join(file_names))  # Column letters again

        # Show last move if available
        if self.board.move_stack:
            last_move = self.board.peek().uci()
            print(f"Last move: {last_move}")
        if use_color:
            print(self._RESET_COLOR, end='')  # Ensure terminal color is reset

    def get_legal_actions(self, state):
        board = self._state_to_board(state)
        legal_moves = [move.uci() for move in board.legal_moves]
        legal_actions = [self.encode_uci_to_action_index(move, board) for move in legal_moves]
        return legal_actions
    
    def encode_uci_to_action_index(self, uci: str, board=None) -> int:
        if board is None:
            board = self.board
        move = chess.Move.from_uci(uci)
        from_square = move.from_square
        to_square = move.to_square
        promotion = move.promotion

        from_file, from_rank = chess.square_file(from_square), chess.square_rank(from_square)
        to_file, to_rank = chess.square_file(to_square), chess.square_rank(to_square)
        
        delta_file, delta_rank = to_file - from_file, to_rank - from_rank
        
        piece = board.piece_at(from_square)
        if piece is None: return -1 # Should not happen for legal move
        
        move_type = -1
        
        # Underpromotion
        if promotion is not None and promotion != chess.QUEEN:
            try:
                promo_idx = self.underpromotion_pieces.index(promotion)
                direction_idx = delta_file + 1 # file delta -1,0,1 -> 0,1,2
                move_type = 64 + promo_idx * 3 + direction_idx
            except (ValueError, IndexError):
                return -1
        
        # Knight move
        elif piece.piece_type == chess.KNIGHT:
            try:
                knight_move_idx = self.knight_move_deltas.index((delta_file, delta_rank))
                move_type = 56 + knight_move_idx
            except ValueError:
                return -1 # Should not happen
                
        # Queen-like move (includes normal pawn moves, king moves, and queen promotions)
        else:
            abs_delta_file, abs_delta_rank = abs(delta_file), abs(delta_rank)
            
            # Not a straight or diagonal line
            if (delta_file != 0 and delta_rank != 0 and abs_delta_file != abs_delta_rank):
                return -1
            
            distance = max(abs_delta_file, abs_delta_rank)
            if distance == 0: return -1

            norm_delta_file = delta_file // distance
            norm_delta_rank = delta_rank // distance
            
            try:
                direction_idx = self.queen_move_directions.index((norm_delta_file, norm_delta_rank))
                move_type = direction_idx * 7 + (distance - 1)
            except ValueError:
                return -1
        
        if move_type == -1:
            return -1
            
        return from_square * 73 + move_type

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

    def _decode_action_index(self, action: int, board=None):
        if board is None:
            board = self.board
        from_square = action // 73
        move_type = action % 73
        from_file, from_rank = chess.square_file(from_square), chess.square_rank(from_square)

        to_square = -1  # Invalid default
        promotion = None

        # 1. Queen-like moves (0-55)
        if 0 <= move_type < 56:
            direction_idx = move_type // 7
            distance = (move_type % 7) + 1
            
            delta_file, delta_rank = self.queen_move_directions[direction_idx]
            to_file = from_file + delta_file * distance
            to_rank = from_rank + delta_rank * distance

            if 0 <= to_file < 8 and 0 <= to_rank < 8:
                to_square = chess.square(to_file, to_rank)
                piece = board.piece_at(from_square)
                if piece and piece.piece_type == chess.PAWN:
                    if (board.turn == chess.WHITE and from_rank == 6 and to_rank == 7) or \
                       (board.turn == chess.BLACK and from_rank == 1 and to_rank == 0):
                        promotion = chess.QUEEN

        # 2. Knight moves (56-63)
        elif 56 <= move_type < 64:
            delta_file, delta_rank = self.knight_move_deltas[move_type - 56]
            to_file = from_file + delta_file
            to_rank = from_rank + delta_rank
            if 0 <= to_file < 8 and 0 <= to_rank < 8:
                to_square = chess.square(to_file, to_rank)

        # 3. Underpromotions (64-72)
        elif 64 <= move_type < 73:
            promo_idx = (move_type - 64) // 3
            direction_idx = (move_type - 64) % 3
            
            promotion = self.underpromotion_pieces[promo_idx]
            
            if board.turn == chess.WHITE:
                to_rank = from_rank + 1
                to_file = from_file + (direction_idx - 1)
            else: # BLACK
                to_rank = from_rank - 1
                to_file = from_file + (direction_idx - 1)
            
            if 0 <= to_file < 8 and 0 <= to_rank < 8:
                to_square = chess.square(to_file, to_rank)

        if to_square == -1:
            return from_square, from_square, None

        return from_square, to_square, promotion

    def action_index_to_uci(self, action_index: int, board=None) -> str:
        from_square, to_square, promotion = self._decode_action_index(action_index, board)
        move = chess.Move(from_square, to_square, promotion=promotion)
        return move.uci()
