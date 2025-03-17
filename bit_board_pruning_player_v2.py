import numpy as np
import numba
import socket
import pickle
import time
from bit_reversi import ReversiBitBoard, count_flips_for_move, apply_move

position_importance = 0.5
piece_importance = 1 - position_importance

postion_weights = np.array([
    [ 1.00, -0.30,  0.30,  0.10,  0.10,  0.30, -0.30,  1.00 ],
    [-0.30, -0.40, -0.10, -0.10, -0.10, -0.10, -0.40, -0.30 ],
    [ 0.30, -0.10,  0.20,  0.05,  0.05,  0.20, -0.10,  0.30 ],
    [ 0.10, -0.10,  0.05,  0.05,  0.05,  0.05, -0.10,  0.10 ],
    [ 0.10, -0.10,  0.05,  0.05,  0.05,  0.05, -0.10,  0.10 ],
    [ 0.30, -0.10,  0.20,  0.05,  0.05,  0.20, -0.10,  0.30 ],
    [-0.30, -0.40, -0.10, -0.10, -0.10, -0.10, -0.40, -0.30 ],
    [ 1.00, -0.30,  0.30,  0.10,  0.10,  0.30, -0.30,  1.00 ]
], dtype=np.float64)

directions = np.array([
    [ 1,  1],
    [ 1,  0],
    [ 1, -1],
    [ 0,  1],
    [ 0, -1],
    [-1,  1],
    [-1,  0],
    [-1, -1]
], dtype=np.int64)

@numba.njit
def evaluate_board(white_bits, black_bits, player, postion_weights):
    """Evaluate the board state using piece difference and positional weights.

    Args:
        white_bits (int): Bitboard for white pieces.
        black_bits (int): Bitboard for black pieces.
        player (int): The player for whom the board is being evaluated (1 for White, -1 for Black).
        postion_weights (np.ndarray): Positional weights for the board.

    Returns:
        float: Evaluation score.
    """
    piece_total = 0
    piece_diff = 0
    postion_score = 0.0
    for x in range(8):
        for y in range(8):
            idx = x * 8 + y
            bit = 1 << idx
            if white_bits & bit:
                piece_total += 1
                piece_diff += 1
                postion_score += postion_weights[x, y]
            elif black_bits & bit:
                piece_total += 1
                piece_diff -= 1
                postion_score -= postion_weights[x, y]

    piece_diff /= 64.0

    if player == -1:
        piece_diff = -piece_diff
        postion_score = -postion_score
    
    if piece_total < 20:
        position_importance = 1.00
        piece_importance = 0.00
    elif piece_total < 50:
        position_importance = 0.75
        piece_importance = 0.25
    else:
        position_importance = 0.00
        piece_importance = 1.00

    # total_piece_weight = piece_total / 64
    # position_importance = 1 - total_piece_weight
    # piece_importance = 1 - position_importance
    
    return piece_importance * piece_diff + position_importance * postion_score

class SearchTimeout(Exception):
    """Custom exception raised when the search time limit is exceeded."""
    pass

def alphabeta(white_bits, black_bits, turn, depth, player, alpha, beta, directions, postion_weights, start_time, time_limit):
    """Perform recursive alpha-beta search to find the best move.

    Args:
        white_bits (int): Bitboard for white pieces.
        black_bits (int): Bitboard for black pieces.
        turn (int): The current player's turn (1 or -1).
        depth (int): Current search depth.
        player (int): The maximizing player (1 or -1).
        alpha (float): Alpha value for pruning.
        beta (float): Beta value for pruning.
        directions (np.ndarray): Array of direction vectors.
        postion_weights (np.ndarray): Positional weight matrix.
        start_time (float): Search start time.
        time_limit (float): Time limit for search.

    Returns:
        tuple: (evaluation score, best move x-coordinate, best move y-coordinate)
    """
    if time.time() - start_time > time_limit:
        raise SearchTimeout("Time limit exceeded")
    if depth == 0:
        score = evaluate_board(white_bits, black_bits, player, postion_weights)
        return score, -1, -1
    if turn == player:
        max_eval = -1e9
        best_move = (-1, -1)
        legal_move_found = False
        for x in range(8):
            for y in range(8):
                flips = count_flips_for_move(white_bits, black_bits, x, y, turn, directions)
                if flips < 0:
                    continue
                legal_move_found = True
                new_white, new_black, _ = apply_move(white_bits, black_bits, x, y, turn, directions)
                eval_score, _, _ = alphabeta(new_white, new_black, -turn, depth - 1, player, alpha, beta, directions, postion_weights, start_time, time_limit)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = (x, y)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            if beta <= alpha:
                break
        if not legal_move_found:
            eval_score, _, _ = alphabeta(white_bits, black_bits, -turn, depth - 1, player, alpha, beta, directions, postion_weights, start_time, time_limit)
            return eval_score, -1, -1
        return max_eval, best_move[0], best_move[1]
    else:
        min_eval = 1e9
        best_move = (-1, -1)
        legal_move_found = False
        for x in range(8):
            for y in range(8):
                flips = count_flips_for_move(white_bits, black_bits, x, y, turn, directions)
                if flips < 0:
                    continue
                legal_move_found = True
                new_white, new_black, _ = apply_move(white_bits, black_bits, x, y, turn, directions)
                eval_score, _, _ = alphabeta(new_white, new_black, -turn, depth - 1, player, alpha, beta, directions, postion_weights, start_time, time_limit)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = (x, y)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            if beta <= alpha:
                break
        if not legal_move_found:
            eval_score, _, _ = alphabeta(white_bits, black_bits, -turn, depth - 1, player, alpha, beta, directions, postion_weights, start_time, time_limit)
            return eval_score, -1, -1
        return min_eval, best_move[0], best_move[1]

def iterative_deepening(white_bits, black_bits, turn, player, time_limit, directions, postion_weights):
    """Perform iterative deepening search within a given time limit.

    Args:
        white_bits (int): Bitboard for white pieces.
        black_bits (int): Bitboard for black pieces.
        turn (int): Current player's turn (1 or -1).
        player (int): The maximizing player.
        time_limit (float): Maximum allowed time for search.
        directions (np.ndarray): Array of direction vectors.
        postion_weights (np.ndarray): Positional weight matrix.

    Returns:
        tuple: (best score, best move as (x, y), final depth reached)
    """
    start_time = time.time()
    best_move = (-1, -1)
    best_score = None
    depth = 1
    while True:
        try:
            score, move_x, move_y = alphabeta(white_bits, black_bits, turn, depth, player, -1e9, 1e9, directions, postion_weights, start_time, time_limit)
            best_score = score
            if move_x != -1:
                best_move = (move_x, move_y)
            depth += 1
        except SearchTimeout:
            print("Time limit reached at depth", depth)
            break
    final_depth = depth - 1
    return best_score, best_move, final_depth

def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    game = ReversiBitBoard()
    TIME_LIMIT = 4.9

    while True:
        data = game_socket.recv(4096)
        if not data:
            break

        turn, board_np = pickle.loads(data)
        if turn == 0:
            print("\n===== Game Over =====")
            game_socket.close()
            return

        print("Current turn:", turn)
        print("Board state:\n", board_np)

        game.set_board_from_numpy(board_np)
        game.turn = turn

        # Use the iterative deepening search with time limit.
        start_search_time = time.time()
        score, move, search_depth = iterative_deepening(
            game.white_bits, game.black_bits, turn, turn, TIME_LIMIT, directions, postion_weights)
        if move is None:
            move = (-1, -1)
        elapsed = time.time() - start_search_time

        print("AlphaBeta selected move:", move,
              "with evaluation score:", score,
              f"(searched for {elapsed:.2f} seconds, depth={search_depth})")
        print(f"Searched up to {search_depth} plies ahead.\n")

        game_socket.send(pickle.dumps(move))

if __name__ == '__main__':
    main()
