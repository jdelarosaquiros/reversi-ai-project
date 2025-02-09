import numpy as np
import socket
import pickle
import random
import math
import time  # Import time module for the timer
from reversi import reversi

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
    moves = []
    # Try every board position
    for i in range(8):
        for j in range(8):
            if game.board[i, j] == 0:
                # Use commit=False so the board is not modified.
                if game.step(i, j, turn, commit=False) > 0:
                    moves.append((i, j))
    if not moves:
        moves.append((-1, -1))
    return moves

def evaluate(game, player):
    """
    Evaluation using the maintained piece counts.
    For player 1 (white), return white_count - black_count;
    for player -1 (black), return black_count - white_count.
    """
    return game.white_count - game.black_count if player == 1 else game.black_count - game.white_count

# --- MCTS Implementation ---

class MCTSNode:
    def __init__(self, game, parent=None, move=None):
        self.game = game               # game state at this node
        self.parent = parent           # parent node in the tree
        self.move = move               # move that led to this game state
        self.children = []             # list of child nodes
        self.wins = 0                  # cumulative reward
        self.visits = 0                # number of times node was visited
        # Untried moves for this state (for the current player's turn)
        self.untried_moves = get_valid_moves(game, game.turn)
    
    def select_child(self):
        """
        Select a child node using the UCT (Upper Confidence bounds applied to Trees) formula.
        """
        C = 1.41  # Exploration constant
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            # UCT formula: (win rate) + C * sqrt( log(parent.visits)/child.visits )
            score = child.wins / child.visits + C * math.sqrt(math.log(self.visits) / child.visits)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child
    
    def expand(self):
        """
        Expand by taking one of the untried moves.
        """
        move = self.untried_moves.pop()  # Choose and remove one untried move
        new_game = clone_game(self.game)
        if move == (-1, -1):
            new_game.turn = -new_game.turn  # simulate pass
        else:
            new_game.step(move[0], move[1], new_game.turn, commit=True)
            new_game.turn = -new_game.turn
        child_node = MCTSNode(new_game, parent=self, move=move)
        self.children.append(child_node)
        return child_node
    
    def update(self, result):
        """
        Update this node's statistics.
        """
        self.visits += 1
        self.wins += result

def rollout(game, player):
    """
    Play a random simulation (rollout) from the current game state until the game ends.
    Return a reward: 1 for win, 0 for loss, 0.5 for draw (from the perspective of 'player').
    """
    current_game = clone_game(game)
    pass_count = 0  # count consecutive passes
    while True:
        moves = get_valid_moves(current_game, current_game.turn)
        # If only a pass move is available, count it.
        if moves == [(-1, -1)]:
            pass_count += 1
            if pass_count == 2:  # both players passed -> game over
                break
            current_game.turn = -current_game.turn
            continue
        else:
            pass_count = 0
            move = random.choice(moves)
            if move == (-1, -1):
                current_game.turn = -current_game.turn
            else:
                current_game.step(move[0], move[1], current_game.turn, commit=True)
                current_game.turn = -current_game.turn
    # Use the provided evaluation function to decide the outcome.
    eval_value = evaluate(current_game, player)
    if eval_value > 0:
        return 1   # win
    elif eval_value < 0:
        return 0   # loss
    else:
        return 0.5  # draw

def mcts(root_game, time_limit, player):
    """
    Perform MCTS search from the given game state until time_limit seconds have elapsed.
    
    Parameters:
      - root_game: the current game state.
      - time_limit: maximum time (in seconds) to run the search.
      - player: the player (1 or -1) from whose perspective we measure wins.
    
    Returns:
      The move (as a tuple) selected by MCTS.
    """
    root_node = MCTSNode(clone_game(root_game))
    start_time = time.time()  # Start the timer
    while time.time() - start_time < time_limit:
        node = root_node
        # --- Selection ---
        # Traverse the tree until reaching a node with untried moves or no children.
        while node.untried_moves == [] and node.children != []:
            node = node.select_child()
        # --- Expansion ---
        if node.untried_moves:
            node = node.expand()
        # --- Simulation (Rollout) ---
        result = rollout(node.game, player)
        # --- Backpropagation ---
        while node is not None:
            node.update(result)
            node = node.parent
    # After the time limit, choose the move of the most-visited child.
    best_child = max(root_node.children, key=lambda n: n.visits) if root_node.children else None
    if best_child is not None:
        print("MCTS stats: visits =", best_child.visits, "wins =", best_child.wins)
        return best_child.move  
    else:
        return (-1, -1)

# --- Main game loop (using MCTS instead of alpha-beta) ---

def main():
    # Connect to the server (make sure the host/port match your server settings)
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    
    # Create a local game instance for simulation.
    game = reversi()
    
    # Set the time limit for MCTS in seconds.
    TIME_LIMIT = 4.9

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

        # Run MCTS search from the current position with a 4.9-second time limit.
        move = mcts(game, TIME_LIMIT, turn)
        if move is None:
            move = (-1, -1)  # if no move is found, pass.

        print("MCTS selected move:", move)

        # Send the selected move back to the server.
        game_socket.send(pickle.dumps(move))

if __name__ == '__main__':
    main()
