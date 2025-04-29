import numpy as np
import socket, pickle
from reversi import reversi
from alpha_zero_general.MCTS import MCTS
from alpha_zero_general.othello.OthelloGame import OthelloGame
from alpha_zero_general.othello.pytorch.NNet import NNetWrapper as NNet
from alpha_zero_general.utils import *

def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    game = reversi()
    
    g = OthelloGame(8)
    nnet = NNet(g)
    nnet.load_checkpoint('./alpha_zero_general/pretrained_models/othello/pytorch/','8x8_rev_best.pth.tar')
    args1 = dotdict({'numMCTSSims': 100, 'cpuct':1.0})
    mcts = MCTS(g, nnet, args1)

    while True:

        #Receive play request from the server
        #turn : 1 --> you are playing as white | -1 --> you are playing as black
        #board : 8*8 numpy array
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        #Turn = 0 indicates game ended
        if turn == 0:
            game_socket.close()
            return
        
        #Debug info
        print(turn)
        print(board)

        # Deep Player
        game.board = board
        
        canon_board = g.getCanonicalForm(game.board, turn)
        probs = mcts.getActionProb(canon_board, temp=0) # turn???????
        print(np.argmax(probs))
        best_move = np.argmax(probs)
        x, y = (int(best_move/8), best_move%8)
        
        valids = g.getValidMoves(canon_board, 1)

        if valids[best_move] == 0:
            x, y = -1, -1
        
        #Send your move to the server. Send (x,y) = (-1,-1) to tell the server you have no hand to play
        game_socket.send(pickle.dumps([x,y]))
        
if __name__ == '__main__':
    main()