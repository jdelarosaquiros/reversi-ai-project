import numpy as np
import socket
import pickle
import time
from reversi import reversi

# Global cache dictionary for transposition table caching.
# The keys will be (board_state, turn, depth, player)
cache = {}

# Custom exception for timeouts.
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
    moves = []  # list will hold tuples: (flip_count, (x, y))
    for i in range(8):
        for j in range(8):
            if game.board[i, j] == 0:
                flipped = game.step(i, j, turn, commit=False)
                if flipped > 0:
                    # Binary insertion so that moves are sorted in descending order.
                    low, high = 0, len(moves)
                    while low < high:
                        mid = (low + high) // 2
                        if moves[mid][0] < flipped:
                            high = mid
                        else:
                            low = mid + 1
                    moves.insert(low, (flipped, (i, j)))
    if not moves:
        moves.append((0, (-1, -1)))  # No legal moves; must pass.
    # Return only the move coordinates.
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

    # Create a hashable key for the current state.
    # Convert the numpy board into a tuple of tuples.
    board_key = tuple(map(tuple, game.board))
    key = (board_key, game.turn, depth, player)
    if key in cache:
        return cache[key]

    if depth == 0:
        result = (evaluate(game, player), None)
        cache[key] = result
        return result

    moves = get_valid_moves(game, game.turn)

    # If the only available move is to pass...
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
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # alpha cutoff
        result = (min_eval, best_move)
        cache[key] = result
        return result

def iterative_deepening(game, player, time_limit):
    """
    Iteratively deepen the search until the time limit is reached.
    Returns the best evaluation score and move found so far.
    """
    global cache
    cache.clear()  # Clear the cache before starting a new move search.
    start_time = time.time()
    best_move = (-1, -1)
    best_score = None
    depth = 1

    # Keep searching with increasing depth until time runs out.
    while True:
        try:
            score, move = alphabeta(game, depth, -float('inf'), float('inf'),
                                      player, start_time, time_limit)
            best_score = score
            best_move = move
            depth += 1
        except Timeout:
            print("Max Depth Reached:", depth)
            break
        # Also check time at the end of an iteration.
        if time.time() - start_time >= time_limit:
            break
    return best_score, best_move

def main():
    # Connect to the server (make sure the host/port match your server settings)
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    
    # Create a local game instance for simulation.
    game = reversi()

    TIME_LIMIT = 4.9  # Maximum search time in seconds.

    while True:
        # Receive a play request from the server.
        # The server sends a pickled list: [turn, board]
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        # A turn value of 0 indicates that the game is over.
        if turn == 0:
            game_socket.close()
            return

        print("Current turn:", turn)
        print("Board state:\n", board)

        # Update our local game state.
        game.board = board
        game.turn = turn

        # Run iterative deepening with a time limit.
        start_search_time = time.time()
        score, move = iterative_deepening(game, turn, TIME_LIMIT)
        if move is None:
            move = (-1, -1)  # if no move is found, pass.

        elapsed = time.time() - start_search_time
        print("AlphaBeta selected move:", move, "with evaluation score:", score, f"(searched for {elapsed:.2f} seconds)")

        # Send the selected move back to the server.
        game_socket.send(pickle.dumps(move))

if __name__ == '__main__':
    main()
