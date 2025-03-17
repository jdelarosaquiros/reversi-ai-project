import subprocess
import time

def main():
    for i in range(20):
        server = subprocess.Popen("python reversi_server_iterative.py", shell=True)
        time.sleep(2)
        player1 = subprocess.Popen("python bit_board_pruning_player.py", shell=True)
        time.sleep(1)
        player2 = subprocess.Popen("python pruning_player.py", shell=True)
        server.wait()

if __name__ == '__main__':
    main()
