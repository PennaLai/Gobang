import numpy as np
from main import AI as AI
from main1 import AI as AI1


def display_chessboard(chessboard):
    colors = ['·', '◯', '●']
    print('\t', end='')
    for i in range(chessboard[0].size):
        print(i, end='\t')
    print()
    start = 0
    for e in chessboard:
        print(start, end=':\t')
        for j in e:
            # if (j == )
            print(colors[int(j)], end='\t')
        start += 1
        print()

# line_processor use for duplicated process of lines in function evaluator
def line_processor(line, color):
    colors = ['-', 'o', 'x']  # o表示自己, x表示对手, 所以这个数组是给白棋用的
    if color == -1:
        colors = ['-', 'x', 'o']
    result = line.tolist()
    result.insert(0, -color)
    result.append(-color)
    return [colors[int(c)] for c in result]


def go_bang(chessboard, position, color):
    assert chessboard[position[0], position[1]] == 0
    chessboard[position[0], position[1]] = color

# 测试用
def AI_go(chessboard, arti):
    arti.go(chessboard)
    # print(ai.candidate_list)
    go_bang(chessboard, arti.candidate_list[-1], arti.color)


def who_win(chessboard):
    length = len(chessboard)
    for i in range(length):
        if kmp_match(line_processor(chessboard[i, :], 1), 'ooooo') > 0 or \
                kmp_match(line_processor(chessboard[:, i], 1), 'ooooo') > 0:
            print("White win, Game over")
            display_chessboard(chessboard)
            return True
        if kmp_match(line_processor(chessboard[i, :], -1), 'ooooo') > 0 or \
                kmp_match(line_processor(chessboard[:, i], -1), 'ooooo') > 0:
            print("Black win, Game over")
            display_chessboard(chessboard)
            return True
    h = [i for i in range(length)]
    v = h[::-1]
    for i in range(5, length + 1):
        if i < length:
            if kmp_match(line_processor(chessboard[h[:i - length], v[length - i:]], 1), 'ooooo') > 0 or \
                    kmp_match(line_processor(chessboard[h[:i - length], h[length - i:]], 1), 'ooooo') > 0:
                print("White win, Game over")
                display_chessboard(chessboard)
                return True
            if kmp_match(line_processor(chessboard[h[:i - length], v[length - i:]], -1), 'ooooo') > 0 or \
                    kmp_match(line_processor(chessboard[h[:i - length], h[length - i:]], -1), 'ooooo') > 0:
                print("Black win, Game over")
                display_chessboard(chessboard)
                return True
    return False


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


# KMP
def kmp_match(s, p):
    m = len(s)  # string
    n = len(p)  # pattern
    count = 0
    cur = 0  # 起始指针cur
    table = partial_table(p)
    while cur <= m - n:
        for i in range(n):
            if s[i + cur] != p[i]:
                cur += max(i - table[i - 1], 1)  # 有了部分匹配表,我们不只是单纯的1位1位往右移,可以一次移动多位
                break
        else:
            count = count + 1
            cur += 1
    return count


# 部分匹配表
def partial_table(p):
    prefix = set()
    ret = [0]
    for i in range(1, len(p)):
        prefix.add(p[:i])
        postfix = {p[j:i + 1] for j in range(1, i + 1)}
        ret.append(len((prefix & postfix or {''}).pop()))
    return ret


if __name__ == '__main__':
    cb = np.zeros([15, 15], dtype=int)
    # cb[14, 5] = COLOR_BLACK
    # cb[5, 5] = COLOR_BLACK
    # no = np.where(cb == COLOR_NONE)
    # played = np.where(cb != COLOR_NONE)
    # played_list = list(zip(played[0], played[1]))
    #
    display_chessboard(cb)
    # ai = aii(15, COLOR_BLACK, 100)
    ai = AI(15, 1, 100)
    ai2 = AI1(15, -1, 100)
    # ai2 = AI(15, COLOR_WHITE, 100)
    x = 0
    y = 0
    while np.where(cb == 0)[0].size > 0 and not who_win(chessboard=cb):
        AI_go(cb, ai2)
        display_chessboard(cb)
        # tempt = input("Enter to continue")
        AI_go(cb, ai)
        display_chessboard(cb)