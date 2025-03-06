import numpy as np
import numba

@numba.njit
def bit_index(x, y):
    """Return the bit index corresponding to board coordinates (x, y)."""
    # 1, 1 => 9
    return x * 8 + y

@numba.njit
def get_bit(board_bits, x, y):
    """Return the bit (0 or 1) at position (x, y) from the given bitboard."""
    return (board_bits >> bit_index(x, y)) & 1

@numba.njit
def set_bit(board_bits, x, y):
    """Set the bit at position (x, y) in the bitboard and return the updated bitboard."""
    return board_bits | (1 << bit_index(x, y))

@numba.njit
def clear_bit(board_bits, x, y):
    """Clear the bit at position (x, y) in the bitboard and return the updated bitboard."""
    return board_bits & ~(1 << bit_index(x, y))

@numba.njit
def count_flips_for_move(white_bits, black_bits, x, y, piece, directions):
    """Count the number of opponent pieces that would be flipped by placing a piece at (x, y).

    Returns:
        int: The number of flips, or -1 if the square is occupied, or -3 if no flips are possible.
    """
    idx = x * 8 + y
    if ((white_bits >> idx) & 1) or ((black_bits >> idx) & 1):
        return -1
    flips = 0
    if piece == 1:
        my_bits = white_bits
        opp_bits = black_bits
    else:
        my_bits = black_bits
        opp_bits = white_bits
    for d in range(directions.shape[0]):
        dx = directions[d, 0]
        dy = directions[d, 1]
        cur_x = x + dx
        cur_y = y + dy
        line_flips = 0
        while 0 <= cur_x < 8 and 0 <= cur_y < 8:
            cur_idx = cur_x * 8 + cur_y
            if (((my_bits >> cur_idx) & 1) == 0) and (((opp_bits >> cur_idx) & 1) == 0):
                break
            if (my_bits >> cur_idx) & 1:
                if line_flips > 0:
                    flips += line_flips
                break
            line_flips += 1
            cur_x += dx
            cur_y += dy
    if flips == 0:
        return -3
    return flips

@numba.njit
def apply_move(white_bits, black_bits, x, y, piece, directions):
    """Apply a move at position (x, y) for the given piece and update the bitboards.
    white_bits = 0000000... 010 & 1
    black_bits = 0011000 ... 10010
    
    Args:
        white_bits (int): Bitboard for white pieces.
        black_bits (int): Bitboard for black pieces.
        x (int): Row index.
        y (int): Column index.
        piece (int): 1 for White, -1 for Black.
        directions (np.ndarray): Array of direction vectors.

    Returns:
        tuple: (updated_white_bits, updated_black_bits, flips) or original boards with negative flips if move is illegal.
    """
    flips = count_flips_for_move(white_bits, black_bits, x, y, piece, directions)
    if flips < 0:
        return white_bits, black_bits, flips
    if piece == 1:
        my_bits = white_bits
        opp_bits = black_bits
    else:
        my_bits = black_bits
        opp_bits = white_bits
    my_bits = set_bit(my_bits, x, y)
    for d in range(directions.shape[0]):
        dx = directions[d, 0]
        dy = directions[d, 1]
        cur_x = x + dx
        cur_y = y + dy
        line_flips = 0
        while 0 <= cur_x < 8 and 0 <= cur_y < 8:
            cur_idx = cur_x * 8 + cur_y
            if (((my_bits >> cur_idx) & 1) == 0) and (((opp_bits >> cur_idx) & 1) == 0):
                break
            if (my_bits >> cur_idx) & 1:
                if line_flips > 0:
                    tx = x + dx
                    ty = y + dy
                    for i in range(line_flips):
                        my_bits = set_bit(my_bits, tx, ty)
                        opp_bits = clear_bit(opp_bits, tx, ty)
                        tx += dx
                        ty += dy
                break
            line_flips += 1
            cur_x += dx
            cur_y += dy
    if piece == 1:
        return my_bits, opp_bits, flips
    else:
        return opp_bits, my_bits, flips

class ReversiBitBoard:
    def __init__(self):
        """Initialize the bit board with the standard starting positions."""
        self.white_bits = 0
        self.black_bits = 0
        self.white_bits = set_bit(self.white_bits, 3, 3)
        self.white_bits = set_bit(self.white_bits, 4, 4)
        self.black_bits = set_bit(self.black_bits, 3, 4)
        self.black_bits = set_bit(self.black_bits, 4, 3)
        self.white_count = 2
        self.black_count = 2
        self.turn = 1

    def get_board_as_numpy(self):
        """Return the board state as an 8x8 NumPy array with 1 for white, -1 for black, and 0 for empty."""
        board = np.zeros((8, 8), dtype=int)
        for x in range(8):
            for y in range(8):
                if get_bit(self.white_bits, x, y):
                    board[x, y] = 1
                elif get_bit(self.black_bits, x, y):
                    board[x, y] = -1
        return board

    def set_board_from_numpy(self, board):
        """Set the board state from an 8x8 NumPy array."""
        self.white_bits = 0
        self.black_bits = 0
        white_count = 0
        black_count = 0
        for x in range(8):
            for y in range(8):
                if board[x, y] == 1:
                    self.white_bits = set_bit(self.white_bits, x, y)
                    white_count += 1
                elif board[x, y] == -1:
                    self.black_bits = set_bit(self.black_bits, x, y)
                    black_count += 1
        self.white_count = white_count
        self.black_count = black_count

    def clone(self):
        """Return a clone of the current board state."""
        new_game = ReversiBitBoard()
        new_game.white_bits = self.white_bits
        new_game.black_bits = self.black_bits
        new_game.white_count = self.white_count
        new_game.black_count = self.black_count
        new_game.turn = self.turn
        return new_game