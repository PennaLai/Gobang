import numpy as np
from main import AI as AI


def display_chessboard(chessboard):
    colors = ['·', '◯', '◉']
    print('\t', end='')
    for i in range(chessboard[0].size):
        print(i, end='\t')
    print()
    start = 0
    for e in chessboard:
        print(start, end=':\t')
        for j in e:
            print(colors[j], end='\t')
        start += 1
        print()


# -1 black, 1 white
def play(chessboard, chess, inChess):
    x, y = inChess
    if x < 0 or x > 14 or y < 0 or y > 14:
        print('越界了重新下')
        play(chessboard, chess)
        return
    if chessboard[x][y] != 0:
        print('当前位置已经满了，请重新下')
        play(chessboard, chess)
        return
    if chess == -1:
        chessboard[x][y] = -1
    if chess == 1:
        chessboard[x][y] = 1


if __name__ == '__main__':
    AI_COLOR = 1
    chessboard = np.zeros([15, 15], dtype=int)
    # start with black chess
    chessNow = -1
    # AI hold while chess
    A = AI(15, AI_COLOR, 15)
    size = A.chessboard_size
    player_step = 0
    while True:
        if chessNow == AI_COLOR:
            A.go(chessboard)
            AI_result = A.get_result()
            play(chessboard, chessNow, AI_result)
        else:
            x, y = input().split()
            x, y = int(x), int(y)
            User_result = (x, y)
            print('对方下在了',User_result)
            play(chessboard, chessNow, User_result)
            # print(evalocal(size, User_result, chessboard, 1))
        display_chessboard(chessboard)
        chessNow = - chessNow