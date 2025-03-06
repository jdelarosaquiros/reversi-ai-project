import numpy as np
import socket
import pickle
import time
from reversi import reversi

################################################
# Global timing and call counters
################################################

time_spent = {
    'get_valid_moves': 0.0,
    'clone_game': 0.0,
    'evaluate': 0.0,
    'alphabeta': 0.0,
    'alphabeta_other': 0.0,
    'alphabeta_all': 0.0,
    'step': 0.0,
    'iterative_deepening': 0.0
}

calls_count = {
    'get_valid_moves': 0,
    'clone_game': 0,
    'evaluate': 0,
    'alphabeta': 0,
    'alphabeta_other': 0,
    'alphabeta_all': 0,
    'step': 0,
    'iterative_deepening': 0
}

weights = [
    [120, -20,  20,   5,   5,  20, -20, 120],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [120, -20,  20,   5,   5,  20, -20, 120],
]
# weights = [
#     [2000000000000, -1000000000000, 0, 0, 0, 0, -1000000000000, 2000000000000],
#     [-1000000000000, -1000000000000, 0, 0, 0, 0, -1000000000000, -1000000000000],
#     [0,             0,              0, 0, 0, 0, 0,              0],
#     [0,             0,              0, 0, 0, 0, 0,              0],
#     [0,             0,              0, 0, 0, 0, 0,              0],
#     [0,             0,              0, 0, 0, 0, 0,              0],
#     [-1000000000000, -1000000000000, 0, 0, 0, 0, -1000000000000, -1000000000000],
#     [2000000000000, -1000000000000, 0, 0, 0, 0, -1000000000000, 2000000000000],
# ]


# A simple helper to measure time and increment call count
def timed(func_name):
    """
    Returns a decorator that measures how much time is spent 
    in a function and how many times it is called.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            calls_count[func_name] += 1
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            time_spent[func_name] += (end_time - start_time)
            return result
        return wrapper
    return decorator

# Global cache dictionary for transposition table caching.
# The keys will be (board_state, turn, depth, player)
cache = {}

@timed('clone_game')
def clone_game(game):
    """
    Create a new reversi instance that is a shallow copy of game.
    We copy the board (using .copy()) and all counters/attributes.
    """
    new_game = reversi()
    new_game.board = game.board.copy()
    new_game.white_count = game.white_count
    new_game.black_count = game.black_count
    new_game.turn = game.turn
    new_game.time = game.time
    return new_game

@timed('get_valid_moves')
def get_valid_moves(game, turn):
    """
    Return a list of legal moves for the given turn.
    Each legal move is inserted in descending order based on the number
    of pieces that would be flipped.

    If no move is legal, return [(-1, -1)] to denote a pass.
    """
    moves = []
    for i in range(8):
        for j in range(8):
            if game.board[i, j] == 0:
                flipped = game.step(i, j, turn, commit=False)
                if flipped > 0:
                    # Binary insertion sort by number of flipped pieces
                    low, high = 0, len(moves)
                    while low < high:
                        mid = (low + high) // 2
                        if moves[mid][0] < flipped:
                            high = mid
                        else:
                            low = mid + 1
                    moves.insert(low, (flipped, (i, j)))
    if not moves:
        moves.append((0, (-1, -1)))
    return [move[1] for move in moves]
# @timed('get_valid_moves')
# def get_valid_moves(game, turn):
#     """
#     Return a list of legal moves for the given turn.
#     If no move is legal, return [(-1, -1)] to denote a pass.
#     """
#     moves = []
#     for i in range(8):
#         for j in range(8):
#             if game.board[i, j] == 0:
#                 flipped = game.step(i, j, turn, commit=False)
#                 if flipped > 0:
#                     moves.append((i, j))
#     if not moves:
#         moves.append((-1, -1))
#     return moves

# @timed('evaluate')
# def evaluate(game, player):
#     """
#     Evaluation using the maintained piece counts.
#     For player 1 (white), return white_count - black_count;
#     for player -1 (black), return black_count - white_count.
#     """
#     if player == 1:  # white
#         return game.white_count - game.black_count
#     else:  # black
#         return game.black_count - game.white_count


@timed('evaluate')
def evaluate(game, player: int):
    # Base piece difference
    piece_diff = game.white_count - game.black_count
    if player == -1:
        piece_diff = -piece_diff

    board = game.board
    # Positional score
    positional_score = 0
    for x in range(8):
        for y in range(8):
            if board[x][y] == 1:
                positional_score += weights[x][y]
            elif board[x][y] == -1:
                positional_score -= weights[x][y]

    # If player == -1, we flip the sign of positional_score
    if player == -1:
        positional_score = -positional_score

    # Combine them, possibly with weights. You can tune these multipliers.
    # For example, if you want position to matter more, multiply by a larger factor.
    return piece_diff + 0.5 * positional_score

class Timeout(Exception):
    pass

@timed('alphabeta')
def alphabeta(game, depth, alpha, beta, player, start_time, time_limit):
    """
    Alpha-beta minimax search with caching and a time check.
    At each node we check whether the allowed time has passed and,
    if so, raise a Timeout exception.
    """
    aba_start = time.time()
    abo_start = time.time()
    if time.time() - start_time > time_limit:
        raise Timeout()

    board_key = tuple(map(tuple, game.board))
    key = (board_key, game.turn, depth, player)
    if key in cache:
        aba_end = time.time()
        time_spent['alphabeta_all'] += (aba_end - aba_start)
        calls_count['alphabeta_all'] += 1
        return cache[key]

    if depth == 0:
        result = (evaluate(game, player), None)
        cache[key] = result
        aba_end = time.time()
        time_spent['alphabeta_all'] += (aba_end - aba_start)
        calls_count['alphabeta_all'] += 1
        return result
    
    abo_end = time.time()
    time_spent['alphabeta_other'] += (abo_end - abo_start)
    calls_count['alphabeta_other'] += 1
    moves = get_valid_moves(game, game.turn)

    if moves == [(-1, -1)]:
        # Pass move
        new_game = clone_game(game)
        new_game.turn = -new_game.turn  # simulate pass
        # Check if the game ends (both players pass)
        if get_valid_moves(new_game, new_game.turn) == [(-1, -1)]:
            result = (evaluate(new_game, player), None)
            cache[key] = result
            aba_end = time.time()
            time_spent['alphabeta_all'] += (aba_end - aba_start)
            calls_count['alphabeta_all'] += 1
            return result
        else:
            aba_end = time.time()
            time_spent['alphabeta_all'] += (aba_end - aba_start)
            calls_count['alphabeta_all'] += 1
            score, _ = alphabeta(new_game, depth - 1, alpha, beta, player, start_time, time_limit)
            aba_start = time.time()
            result = (score, (-1, -1))
            cache[key] = result
            aba_end = time.time()
            time_spent['alphabeta_all'] += (aba_end - aba_start)
            calls_count['alphabeta_all'] += 1
            return result

    if game.turn == player:
        max_eval = -float('inf')
        best_move = None
        for move in moves:
            new_game = clone_game(game)
            if move == (-1, -1):
                new_game.turn = -new_game.turn  # pass
            else:
                # measure 'step' time
                st_start = time.time()
                new_game.step(move[0], move[1], new_game.turn, commit=True)
                st_end = time.time()
                time_spent['step'] += (st_end - st_start)
                calls_count['step'] += 1

                new_game.turn = -new_game.turn

            aba_end = time.time()
            time_spent['alphabeta_all'] += (aba_end - aba_start)
            calls_count['alphabeta_all'] += 1
            eval_score, _ = alphabeta(new_game, depth - 1, alpha, beta, player, start_time, time_limit)
            aba_start = time.time()
            
            abo_start = time.time()
            # Adjust corners and near-corner heuristics
            # if move in [(0, 0), (7, 7), (0, 7), (7, 0)]:
            #     eval_score += 2_000_000_000_000
            # if move in [(0, 1), (1, 0), (1, 1),
            #             (7, 1), (6, 1), (6, 0),
            #             (0, 6), (1, 6), (1, 7),
            #             (7, 6), (6, 6), (6, 7)]:
            #     eval_score -= 1_000_000_000_000

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, max_eval)
            if beta <= alpha:
                abo_end = time.time()
                time_spent['alphabeta_other'] += (abo_end - abo_start)
                calls_count['alphabeta_other'] += 1
                break
            abo_end = time.time()
            time_spent['alphabeta_other'] += (abo_end - abo_start)
            calls_count['alphabeta_other'] += 1

        result = (max_eval, best_move)
        cache[key] = result
        aba_end = time.time()
        time_spent['alphabeta_all'] += (aba_end - aba_start)
        calls_count['alphabeta_all'] += 1
        return result
    else:
        min_eval = float('inf')
        best_move = None
        for move in moves:
            new_game = clone_game(game)
            if move == (-1, -1):
                new_game.turn = -new_game.turn
            else:
                # measure 'step' time
                st_start = time.time()
                new_game.step(move[0], move[1], new_game.turn, commit=True)
                st_end = time.time()
                time_spent['step'] += (st_end - st_start)
                calls_count['step'] += 1

                new_game.turn = -new_game.turn

            aba_end = time.time()
            time_spent['alphabeta_all'] += (aba_end - aba_start)
            calls_count['alphabeta_all'] += 1
            eval_score, _ = alphabeta(new_game, depth - 1, alpha, beta, player, start_time, time_limit)
            aba_start = time.time()
            
            abo_start = time.time()
            # Adjust corners and near-corner heuristics (opposite sign)
            # if move in [(0, 0), (7, 7), (0, 7), (7, 0)]:
            #     eval_score -= 2_000_000_000_000
            # if move in [(0, 1), (1, 0), (1, 1),
            #             (7, 1), (6, 1), (6, 0),
            #             (0, 6), (1, 6), (1, 7),
            #             (7, 6), (6, 6), (6, 7)]:
            #     eval_score += 1_000_000_000_000

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, min_eval)
            if beta <= alpha:
                abo_end = time.time()
                time_spent['alphabeta_other'] += (abo_end - abo_start)
                calls_count['alphabeta_other'] += 1
                break

            abo_end = time.time()
            time_spent['alphabeta_other'] += (abo_end - abo_start)
            calls_count['alphabeta_other'] += 1

        result = (min_eval, best_move)
        cache[key] = result
        aba_end = time.time()
        time_spent['alphabeta_all'] += (aba_end - aba_start)
        calls_count['alphabeta_all'] += 1
        return result

@timed('iterative_deepening')
def iterative_deepening(game: reversi, player: int, time_limit: float):
    """
    Iteratively deepen the search until the time limit is reached.
    Returns the best evaluation score, best move, and the maximum depth reached.
    """
    global cache
    cache.clear()
    start_time = time.time()
    best_move = (-1, -1)
    best_score = None
    depth = 1

    while True:
        try:
            score, move = alphabeta(
                game, depth,
                -float('inf'), float('inf'),
                player,
                start_time,
                time_limit
            )
            best_score = score
            best_move = move
            depth += 1

        except Timeout:
            print("Max Depth Reached (due to timeout):", depth)
            break

        # Check time at the end of each iteration
        if time.time() - start_time >= time_limit:
            break

    final_depth = depth - 1
    return best_score, best_move, final_depth

def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    
    game = reversi()

    TIME_LIMIT = 4.9

    while True:
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)
        
        if turn == 0:
            # Print final timing statistics when the game ends
            print("\n===== TIMING & CALL STATISTICS =====")
            for func_name, t_spent in time_spent.items():
                print(f"{func_name}: {t_spent:.4f} seconds, called {calls_count[func_name]} times")
            game_socket.close()
            return

        print("Current turn:", turn)
        print("Board state:\n", board)

        game.board = board
        game.turn = turn

        start_search_time = time.time()
        score, move, search_depth = iterative_deepening(game, turn, TIME_LIMIT)
        if move is None:
            move = (-1, -1)

        elapsed = time.time() - start_search_time
        print("AlphaBeta selected move:", move,
              "with evaluation score:", score,
              f"(searched for {elapsed:.2f} seconds, depth={search_depth})")

        print(f"Searched up to {search_depth} moves (plies) ahead.")
        # Print final timing statistics when the game ends
        print("\n===== TIMING & CALL STATISTICS =====")
        for func_name, t_spent in time_spent.items():
            print(f"{func_name}: {t_spent:.4f} seconds, called {calls_count[func_name]} times")

        game_socket.send(pickle.dumps(move))


if __name__ == '__main__':
    main()

