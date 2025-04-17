import numpy as np
import socket
import pickle
import time
from reversi import reversi

# Global cache dictionary for transposition table caching.
# The keys will be (board_state, turn, depth, player)
cache = {}

class Timeout(Exception):
    pass

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
                    # Binary insertion sort
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

def evaluate(game, player):
    """
    Evaluation using the maintained piece counts.
    For player 1 (white), return white_count - black_count;
    for player -1 (black), return black_count - white_count.
    """
    return game.white_count - game.black_count if player == 1 else game.black_count - game.white_count

def alphabeta(game, depth, alpha, beta, player, start_time, time_limit):
    """
    Alpha-beta minimax search with caching and a time check.
    At each node we check whether the allowed time has passed and,
    if so, raise a Timeout exception.
    """
    if time.time() - start_time > time_limit:
        raise Timeout()

    board_key = tuple(map(tuple, game.board))
    key = (board_key, game.turn, depth, player)
    if key in cache:
        return cache[key]

    if depth == 0:
        result = (evaluate(game, player), None)
        cache[key] = result
        return result

    moves = get_valid_moves(game, game.turn)

    if moves == [(-1, -1)]:
        new_game = clone_game(game)
        new_game.turn = -new_game.turn  # simulate pass
        if get_valid_moves(new_game, new_game.turn) == [(-1, -1)]:
            result = (evaluate(game, player), None)
            cache[key] = result
            return result
        else:
            score, _ = alphabeta(new_game, depth - 1, alpha, beta, player, start_time, time_limit)
            result = (score, (-1, -1))
            cache[key] = result
            return result

    if game.turn == player:
        max_eval = -float('inf')
        best_move = None
        for move in moves:
            new_game = clone_game(game)
            if move == (-1, -1):
                new_game.turn = -new_game.turn  # pass move
            else:
                new_game.step(move[0], move[1], new_game.turn, commit=True)
                new_game.turn = -new_game.turn
            eval_score, _ = alphabeta(new_game, depth - 1, alpha, beta, player, start_time, time_limit)
            # Update the eval_score based on position of piece placed
            # set super high score if space is corner
            if move in [(0, 0), (7, 7), (0, 7), (7, 0)]:
                eval_score += 2000000000000
            # set super low score if space is right next to a corner
            if move in [(0, 1), (1, 0), (1, 1), (7, 1), (6, 1), (6,0), (0,6), (1,6), (1,7), (7,6), (6,6), (6,7)]:
                eval_score -= 1000000000000
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, max_eval)
            if beta <= alpha:
                break  # beta cutoff
        result = (max_eval, best_move)
        cache[key] = result
        return result
    else:
        min_eval = float('inf')
        best_move = None
        for move in moves:
            new_game = clone_game(game)
            if move == (-1, -1):
                new_game.turn = -new_game.turn
            else:
                new_game.step(move[0], move[1], new_game.turn, commit=True)
                new_game.turn = -new_game.turn
            eval_score, _ = alphabeta(new_game, depth - 1, alpha, beta, player, start_time, time_limit)
            # Update the eval_score based on position of piece placed
            # Signs are OPPOSITE because this is other opponent player
            # set super low score if space is corner
            if move in [(0, 0), (7, 7), (0, 7), (7, 0)]:
                eval_score -= 2000000000000
            # set super high score if space is right next to a corner
            if move in [(0, 1), (1, 0), (1, 1), (7, 1), (6, 1), (6,0), (0,6), (1,6), (1,7), (7,6), (6,6), (6,7)]:
                eval_score += 1000000000000
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # alpha cutoff
        result = (min_eval, best_move)
        cache[key] = result
        return result


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

        # Also check time at the end of each iteration.
        if time.time() - start_time >= time_limit:
            break

    # Because 'depth' was incremented after a successful search,
    # the last fully-completed depth is 'depth - 1'.
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

        # You can also explicitly print something like:
        print(f"Searched up to {search_depth} moves (plies) ahead.")

        game_socket.send(pickle.dumps(move))

if __name__ == '__main__':
    main()
