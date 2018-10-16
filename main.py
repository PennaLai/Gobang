import numpy as np
import random
import time
import copy

COLOR_BLACK=-1
COLOR_WHITE=1
COLOR_NONE=0
random.seed(0)

# 判断胜负
flag = True
# 评估棋形分值，优先度，0代表空  1 代表我， -1代表对手 ，这种要反转来看，因为从左到右和右到左都要看，但是因为有的棋形是对称的，所以给他第三个值来判断是否对称，是就不用反转了, 第四个值用来记录特殊情况，比如双活三
score_judge = [
               (10, [-1, 1, 1, 1, 1, -1], 1, 0),
               (10, [-1, 1, 1, 1, -1], 1, 0),
               (10, [-1, 1, 1, -1], 1, 0),

               (10, [0, 0, 0, 1, 1, -1], 0, 0),
               (10, [0, 0, 1, 0, 1, -1], 0, 0),
               (10, [0, 1, 0, 0, 1, -1], 0, 0),
               (10, [1, 0, 0, 0, 1], 1, 0),
               (10, [-1, 0, 1, 0, 1, 0, -1], 1, 0),
               (10, [-1, 0, 1, 1, 0, 0, -1], 0, 0),

               (100, [0, 1, 0, 1, 0], 1, 0),
               (100, [0, 0, 1, 1, 0], 0, 0),
               (100, [0, 1, 0, 0, 1, 0], 1, 0),

               (100, [0, 0, 1, 1, 1, -1], 0, 0),
               (100, [0, 1, 0, 1, 1, -1], 0, 0),
               (100, [0, 1, 1, 0, 1, -1], 0, 0),
               (100, [1, 0, 0, 1, 1], 0, 0),
               (100, [1, 0, 1, 0, 1], 1, 0),
               (100, [-1, 0, 1, 1, 1, 0, -1], 1, 0),

               (1000, [0, 1, 0, 1, 1, 0], 0, 1),
               (1000, [0, 1, 1, 1, 0], 1, 1),

               (1000, [0, 1, 1, 1, 1, -1], 0, 2),
               (1000, [0, 1, 0, 1, 1, 1, 0], 0, 2),
               (1000, [0, 1, 1, 0, 1, 1, 0], 1, 2),

               (10000, [0, 1, 1, 1, 1, 0], 1, 0),

               (100000, [1, 1, 1, 1, 1], 1, 0)]

# 分数从高到低分别为 成五，活四，冲四，活三, 眠三，活二，眠二，死四，死三，死二
# 第四个值为 0表示没有什么特殊的，1表示是活三 2表示冲四


class AI(object):

    def __init__(self, clessboard_size, color, time_out):
        self.chessboard_size = clessboard_size
        self.color = color
        # the max time that you can consider
        self.time_out = time_out
        # add decision into candidate_list
        self.candidate_list = []
        # 用来放已经下了棋的节点
        self.played_list = []
        # 每个点的得分记录（初始化中心最高）
        self.chess_mark = self.mark_init()
        # 所有可以下的节点，也就是有邻居的点，并且对分值从高到低排序，一开始为空，每下一个子更新周边八个的分值，并将周边两层非邻居的节点加入邻居list
        if color == -1:
            a = int((self.chessboard_size)/2)
            self.neighbor = [(a, a)] #如果我们先开始，确认一个位置(正常来说，如果是我们开始，第一步一定要下中间
        else:
            self.neighbor = []
        tempt = np.where(np.zeros([self.chessboard_size, self.chessboard_size], dtype=int) == 0)
        self.field = set(list(zip(tempt[0], tempt[1])))

    # 初始化棋盘得分表
    def mark_init(self):
        size = self.chessboard_size
        mark = np.zeros([size, size], dtype=list)
        ma = int((size+1)/2)
        for i in range(ma):
            for j in range(i, size-i):
                # 第一个位置放的是这个点下黑子的评分，第二个是白子的
                mark[i][j] = [i+1, i+1]
                mark[j][i] = [i+1, i+1]
                mark[size-1-i][j] = [i+1, i+1]
                mark[j][size-1-i] = [i+1, i+1]
        return mark

    # 更新下的棋，找出对手下的节点,此时played_list最后一位就是对手下的点
    def update_played_list(self, chessboard):
        idx = np.where(chessboard != COLOR_NONE)
        idx = list(zip(idx[0], idx[1]))
        for i in idx:
            if i not in self.played_list:
                self.played_list.append(i)

    # 把AI（自己）的决策加入到下过的点中
    def add_played_list(self, pos):
        self.played_list.append(pos)

    # current input in the chessboard
    def go(self, chessboard):
        # clear candidate_list
        self.candidate_list.clear()
        # ===============algorithm here================
        # 更新对手下棋位置
        # idx = np.where(chessboard == COLOR_NONE)
        # idx = list(zip(idx[0], idx[1]))
        # pos_idx =random.randint(0, len(idx)-1)
        # new_pos = idx[pos_idx]
        self.update_played_list(chessboard)
        # 对手下了棋后更新数据
        if len(self.played_list) > 0:
            position = self.played_list[-1]
            update_neighbor(position, self.neighbor, chessboard, self.field)
            update_mark(position, chessboard, self.chess_mark, self.field, self.color)
            sort_neighbor(self.chess_mark, self.neighbor, self.color)
        # 开始下棋
        new_pos = self.neighbor[-1]  # 简单的一层搜索
        # 下完棋更新信息
        self.add_played_list(new_pos)
        # print('mark=', self.chess_mark[new_pos[0], new_pos[1]])
        # print('the AI penna go', new_pos, 'and mark= ', evalocal(self.chessboard_size, new_pos, chessboard,self.color))
        update_neighbor(new_pos, self.neighbor, chessboard, self.field)
        update_mark(new_pos, chessboard, self.chess_mark, self.field, self.color)
        sort_neighbor(self.chess_mark, self.neighbor, self.color)
        # 最后做出决策后更新信息

        # =============Find new pos(new pos is a list that contain (x, y)===================================
        # if the position of decision is not empty, return error
        assert chessboard[new_pos[0], new_pos[1]] == COLOR_NONE
        # add decision into candidate_list records the chess board
        self.candidate_list.append(new_pos)


    # get the last decision
    def get_result(self):
        return self.candidate_list[-1]


# 下了棋后，更新评估表的分数，中心包括自己加周边4个, 要传入评估表,
def update_mark(position, chessboard, mark, field, color_main):
    hori = np.arange(position[0] - 4, position[0] + 5, 1)
    vert = np.arange(position[1] - 4, position[1] + 5, 1)
    lens = len(chessboard)
    for h in hori:
        for v in vert:
            tempt = (h, v)
            if tempt in field and chessboard[tempt[0], tempt[1]] == 0:  # 只对空的点评估才有意义， 已经有的点肯定不在neighbor里面
                chessboard[tempt[0], tempt[1]] = color_main
                score = evalocal(lens, (h, v), chessboard, color_main)
                chessboard[tempt[0], tempt[1]] = -color_main
                score1 = evalocal(lens, (h, v), chessboard, -color_main)
                chessboard[tempt[0], tempt[1]] = 0  # 刚刚只是假设，最后要还原
                if color_main == 1:  # 是个白子
                    mark[tempt[0], tempt[1]] = [score1, score] # 第一个放黑子的分数，第二个放白子的分数
                else: # 是个黑子
                    mark[tempt[0], tempt[1]] = [score, score1]
    mark[position[0], position[1]] = [-1, -1]  # 下过的点分数更新为-1，以后不会再用到
    display_mark(mark)


# 每当自己或者对手下了棋，更新neighbor的个数（包括减去已下的位置和更新加入周边的neighbor），要传入neighbor表
def update_neighbor(position, neighbor_list, chessboard, field):
    hori = np.arange(position[0] - 2, position[0] + 3, 1)
    vert = np.arange(position[1] - 2, position[1] + 3, 1)
    for h in hori:
        for v in vert:
            tempt = (h, v)
            if tempt in field:
                if chessboard[tempt[0], tempt[1]] == 0 and tempt not in neighbor_list :
                    neighbor_list.append(tempt)
    for e in neighbor_list:
        if chessboard[e[0], e[1]] != 0 or e[0] == position[0] and e[1] == position[1]: #在更新的时候要把当前下的子给去了
            neighbor_list.remove(e)
    print(neighbor_list)

# 当前棋局评估总分(我的分数减去对面的分数), color是我
def evaluator_total(size, chessboard, color):
    return evaluator(size, chessboard, color) - evaluator(size, chessboard, -color)


# 评估棋盘某个颜色总分数, 墙也算对手
def evaluator(size, chessboard, color):
    total_score = 0
    live_three = 0
    dead_four = 0
    a = 0
    b = 0
    c = 0
    # 对行和列进行评估
    row_sp = np.vsplit(chessboard, size)
    col_sp = np.hsplit(chessboard, size)
    for i in range(size):
        row = [-color]
        col = [-color]
        for j in range(size):
            row.append(row_sp[i][0][j])
            col.append(col_sp[i][j][0])
        total_score += one_row(col, color)
        total_score += one_row(row, color)
    # 对斜的进行评估
    for i in range(size-5+1):
        m = i
        n = size-i-1
        # 假设墙就是对方的阻挡
        left1_incline = [-color]
        left2_incline = [-color]
        right1_incline = [-color]
        right2_incline = [-color]
        for j in range(size-i):
            left1_incline.append(chessboard[m][j])
            left2_incline.append(chessboard[j][m])
            right1_incline.append(chessboard[m][size - j - 1])
            right2_incline.append(chessboard[j][n])
            m += 1
            n -= 1
        left1_incline.append(-color)
        left2_incline.append(-color)
        right1_incline.append(-color)
        right2_incline.append(-color)
        if i == 0:  # 两两重叠多余, 只算其中一个(m 经过+1了)
            a1, b1, c1 = one_row(left1_incline, color)
            a2, b2, c2 = one_row(right1_incline, color)
            total_score = total_score+ a1+a2
            live_three = live_three+ b1+b2
            dead_four += dead_four+c1+c2
            continue
        else:
            a1, b1, c1 = one_row(left1_incline, color)
            a2, b2, c2 = one_row(left2_incline, color)
            a3, b3, c3 = one_row(right1_incline, color)
            a4, b4, c4 = one_row(right2_incline, color)
            total_score = total_score+a1+a2+a3+a4
            live_three = live_three+b1+b2+b3+b4
            dead_four = dead_four + c1+c2+c3+c4
    # 双三那几种情况分开来算
    if live_three >= 2 or live_three + dead_four >= 2 or dead_four >= 2:
        total_score += 10000
    else:
        total_score += live_three * 1000
        total_score += dead_four * 1000
    return total_score


# 启发估值，只估值当前落子中心向外4个位置，返回一个值
def evalocal(size, position, chessboard, color):
    x, y = position
    total_score = 0
    row = []
    col = []
    left_inc = []
    right_inc = []
    live_three = 0  # 活三数量
    dead_four = 0  # 冲四数量
    for i in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
        # row
        if x+i < 0:
            if x+i == -1:
                row.append(-color)
        elif x+i > size-1:
            if x+i == size:
                row.append(-color)
        else:
            row.append(chessboard[x+i][y])
        # col
        if y+i < 0:
            if y+i == -1:
                col.append(-color)
        elif y+i > size-1:
            if y+i == size:
                col.append(-color)
        else:
            col.append(chessboard[x][y+i])
        # left_inc
        if x+i < 0 or y+i < 0:
            if x+i == -1 or y+i == -1:
                left_inc.append(-color)
        elif y+i > size-1 or x+i> size-1:
            if y+i == size or x+i == size:
                left_inc.append(-color)
        else:
            left_inc.append(chessboard[x+i][y+i])
        # right_inc
        if x+i < 0 or y-i > size-1:
            if x+i == -1 or y-i == size:
                right_inc.append(-color)
        elif x+i > size-1 or y-i < 0:
            if x+i == size or y-i == -1:
                right_inc.append(-color)
        else:
            right_inc.append(chessboard[x+i][y-i])
        a, b, c = one_row(row, color)
        total_score += a
        live_three += b
        dead_four += c
        a, b, c = one_row(col, color)
        total_score += a
        live_three += b
        dead_four += c
        a, b, c = one_row(left_inc, color)
        total_score += a
        live_three += b
        dead_four += c
        a, b, c = one_row(right_inc, color)
        total_score += a
        live_three += b
        dead_four += c
    # 双三那几种情况分开来算
    if live_three >= 2 or live_three + dead_four >= 2 or dead_four >= 2:
        print(position, '3=',live_three, '4=',dead_four)
        total_score += 10000
    else:
        total_score += live_three * 1000
        total_score += dead_four * 1000
    return max(10, total_score)

# 根据得分表对neighbor从小到大排序, 这取决于你是白子还是黑子， 你的颜色的分数是1.3倍（因为现在是假设你要下，虽然要考虑对方的分数）
def sort_neighbor(mark_table, neightbor_list, color):
    # 0 是黑色的， 1 是白色的, 这个颜色只用于在层数递归的时候把自己当作对手考虑
    neightbor_list.sort(key=lambda x: mark_table[x[0], x[1]][0] + 1.3 * mark_table[x[0], x[1]][1] if color else 1.3 * mark_table[x[0], x[1]][0]+mark_table[x[0], x[1]][1])

# Alpha-Beta 函数搜索
# def alphabeta(position, neighbor, chessboard, mark_table, depth, maxmin_player, alpha, beta, field, color):
#
#     # 分数到达一定，比如赢了那么大的分数，或者深度到了，就返回
#     if depth == 0 or almost_score > 10000000:
#         return evaluator chessboard value
#
#     if maxmin_player:
#         bes_value = -9999999999999 # 无穷小
#         for pos in neighbor:
#             # 深拷贝， 不影响原来数据，因为大家都要用
#             new_neighbor = copy.deepcopy(neighbor)
#             new_chessboard = copy.deepcopy(chessboard)
#             new_mark_table = copy.deepcopy(mark_table)
#             # 假设要下这个位置，更新下棋盘，评分表， neighbor，然后继续递归
#             update_mark(pos,new_chessboard,new_mark_table, field, color)
#             update_neighbor(pos,new_neighbor,new_chessboard, field)
#             v = alphabeta(pos, new_neighbor, new_chessboard, new_mark_table, depth -1 ,False, alpha, beta)
#             best_value = max(best_value, v)
#             alpha = max(alpha, best_value)
#             if beta <= alpha:
#                 break
#         return best_value
#     else: #考虑对手层
#         best_value = 9999999999999 # 无穷大
#         for pos in neighbor:
#             # 深拷贝， 不影响原来数据，因为大家都要用
#             new_neighbor = copy.deepcopy(neighbor)
#             new_chessboard = copy.deepcopy(chessboard)
#             new_mark_table = copy.deepcopy(mark_table)
#             # 假设要下这个位置，更新下棋盘，评分表， neighbor，然后继续递归
#             update_mark(pos, new_chessboard, new_mark_table, field, -color)
#             update_neighbor(pos, new_neighbor, new_chessboard, field)
#             v = alphabeta(pos, new_neighbor, new_chessboard, new_mark_table, depth - 1, True, alpha, beta)
#             best_value =min(best_value, v)
#             beta = min(beta, best_value)
#             if beta <= alpha :
#                 break
#         return  best_value


# 让估值棋形换成黑棋估值
def neg(a):
    return -a


# 抽离出来的，对一行进行评估, i代表棋形的index, row是待评估的行，返回一个总分， 注意这里只是对一行，所以不能知道整个棋盘的活三数量
def one_row(row, color):
    total_score = 0
    live_three = 0
    death_four = 0
    for i in range(len(score_judge)):  # 所有的棋形
        cas = score_judge[i][1]  # 棋形
        score = score_judge[i][0]  # 分值
        case = score_judge[i][3]  # 样例，是特殊的冲四活三还是什么
        if color == -1:  # 更换评分成黑子的匹配模式
            cas = list(map(neg, cas))
        for inv in range(2):
            if inv == 1:
                if score_judge[i][2] == 1:
                    continue
                else:
                    cas.reverse()  # 只有非对称的才要转过来再算一遍
            time = KMP(row, cas)  # 匹配次数
            if case == 0:
                total_score += time * score
            # 遇到活三冲四先不计分，留给上一层加倍处理
            if case == 1:
                live_three += time
            elif case == 2:
                death_four += time
    return total_score, live_three, death_four


def KMP(st,P) -> int:
    if len(st) < len(P):
        return 0
    nt = nextval(P)
    s, q, k=0, 0, 0# s,q是当前匹配位置,k是匹配开始位置,都是从0开始
    count = 0
    while k < len(st)-len(P)+1:
        q, s = 0, k
        while q < len(P) and P[q] == st[s]:
            q += 1
            s += 1
        # 发生失配或者匹配成功
        if q == len(P): # 匹配成功不要停，继续
            q = 0
            k = k+1
            count += 1
        elif q == 0:  # 一个也没匹配上
            k += 1
        else:  # q是已经匹配的个数
            k += q-nt[q-1]
    return count  # 遍历到最后的k结束后返回count


# next value
def nextval(P):
    #字符的前缀函数
    nt=[0]#nt[]表示P直到下标i的一个偏移(及P(nt[i]-1)是P(i)真前缀的同时也是他的后缀,nt[i]是其长度)
    for i in range(1,len(P)):
        if P[i] == P[nt[i-1]]:#新增的字符可以根据之前的前缀扩张,
            k=nt[i-1]+1
        else:               #新增的字符不能扩张前缀,因此在现有的前缀中找一个更小的前缀
            k=nt[i-1]     #在P(nt[i-1]-1)内找一个小的前缀
            while (P[i]!=P[k] and k!=0):
                k=nt[nt[k-1]]
        nt.append(k)
    return nt


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

def display_mark(mark):
    print('\t', end='')
    for i in range(chessboard[0].size):
        print(i, end='\t')
    print()
    start = 0
    for e in mark:
        print(start, end=':\t')
        for j in e:
            print(j, end='\t')
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
    chessboard = np.zeros([15, 15], dtype=int)
    # start with black chess
    chessNow = -1
    # AI hold while chess
    A = AI(15, -1, 15)
    size = A.chessboard_size
    flag = -1
    while True:
        if chessNow == -1:
            A.go(chessboard)
            AI_result = A.get_result()
            play(chessboard, chessNow, AI_result)
        else:
            x, y = input().split()
            x, y = int(x), int(y)
            User_result = (x, y)
            play(chessboard, chessNow, User_result)
            print('user-mark', evalocal(size, User_result, chessboard, 1))
        display_chessboard(chessboard)
        chessNow = -chessNow
