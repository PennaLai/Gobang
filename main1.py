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
                # 死亡棋局
               (-200, [-1, 1, 1, 1, 1, -1], 1, 0),
               (-200, [-1, 1, 1, 1, -1], 1, 0),
               (-200, [-1, 1, 1, -1], 1, 0),
                # 眠二 落子成眠三
               (10, [0, 0, 0, 1, 1, -1], 0, 0),
               (10, [0, 0, 1, 0, 1, -1], 0, 0),
               (10, [0, 1, 0, 0, 1, -1], 0, 0),
               (10, [1, 0, 0, 0, 1], 1, 0),
               (10, [-1, 0, 1, 0, 1, 0, -1], 1, 0),
               (10, [-1, 0, 1, 1, 0, 0, -1], 0, 0),
                # 活二 落子能成活三
               (100, [0, 1, 0, 1, 0], 1, 0),
               (100, [0, 0, 1, 1, 0], 0, 0),
               (100, [0, 1, 0, 0, 1, 0], 1, 0),
                # 眠三，落子能成冲四
               (100, [0, 0, 1, 1, 1, -1], 0, 0),
               (100, [0, 1, 0, 1, 1, -1], 0, 0),
               (100, [0, 1, 1, 0, 1, -1], 0, 0),
               (100, [1, 0, 0, 1, 1], 0, 0),
               (100, [1, 0, 1, 0, 1], 1, 0),
               (100, [-1, 0, 1, 1, 1, 0, -1], 1, 0),
                # 活三
               (1150, [0, 1, 0, 1, 1, 0], 0, 1),
               (1150, [0, 1, 1, 1, 0, 0], 0, 1), # 这个很特殊，虽然一行可能回重复判断，但是下面写了最多返回一个双三
                # 冲四
               (1100, [0, 1, 1, 1, 1, -1], 0, 2),
               (1100, [1, 0, 1, 1, 1], 0, 2),
               (1100, [1, 1, 0, 1, 1], 1, 2),
                # 一条线上的双冲四
               (12000, [1, 0, 1, 1, 1, 0, 1], 1, 0),
                # 活四 必胜
               (16000, [0, 1, 1, 1, 1, 0], 1, 0),
                # 成五
               (1500000, [1, 1, 1, 1, 1], 1, 0)]

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
        mark = np.zeros([size, size], dtype=int)
        ma = int((size+1)/2)
        for i in range(ma):
            for j in range(i, size-i):
                mark[i][j] = i+1
                mark[j][i] = i+1
                mark[size-1-i][j] = i+1
                mark[j][size-1-i] = i+1
        return mark

    # 更新下的棋，找出对手下的节点,此时played_list最后一位就是对手下的点
    def update_played_list(self, chessboard):
        idx = np.where(chessboard != COLOR_NONE)
        idx = list(zip(idx[0], idx[1]))
        for i in idx:
            if i not in self.played_list:
                self.played_list.append(i)
        return len(self.played_list)
    # 把AI（自己）的决策加入到下过的点中
    def add_played_list(self, pos):
        self.played_list.append(pos)

    # current input in the chessboard
    def go(self, chessboard):
        # clear candidate_list
        self.candidate_list.clear()
        # ===============algorithm here================
        # 更新对手下棋位置, 以及记录当前位置
        step = len(self.played_list)  # 一开始是零
        has_step = self.update_played_list(chessboard) # 更新过的playlist
        position = ()
        # 对手下了棋后更新数据
        if step == 0 and has_step > 0:  # 处理残局
            if step < has_step:
                while step < has_step:
                    pos = self.played_list[step]
                    update_neighbor(pos, self.neighbor, chessboard, self.field)
                    update_mark(pos, chessboard, self.chess_mark, self.field, self.color)
                    sort_neighbor(self.chess_mark, self.neighbor)
                    step += 1
                position = self.played_list[-1]
        elif len(self.played_list) > 0:
            position = self.played_list[-1]
            update_neighbor(position, self.neighbor, chessboard, self.field)
            update_mark(position, chessboard, self.chess_mark, self.field, self.color)
            sort_neighbor(self.chess_mark, self.neighbor)
        # 开始下棋
        # new_pos = self.neighbor[-1]  # 假设我们要下在这, 也就是最小最大的第一个假设
        # print_neighbor_mark(self.chess_mark,self.neighbor)
        # print('Penna选择了',new_pos,'分数是', self.chess_mark[new_pos[0],new_pos[1]])

        # self.neighbor.pop(-1)
        value, new_pos = alphabeta(position, self.neighbor, chessboard, self.chess_mark, 2, self.color, -99999999,999999990, self.field, self.color)  # 一定要返回一个位置的值
        # print('the AI penna go',new1_pos)
        # 下完棋更新信息
        self.add_played_list(new_pos)
        # 因为系统是在我们这个function结束才加入棋子的，所以我们提前加
        chessboard[new_pos[0], new_pos[1]] = self.color
        update_neighbor(new_pos, self.neighbor, chessboard, self.field)
        update_mark(new_pos, chessboard, self.chess_mark, self.field, self.color)
        sort_neighbor(self.chess_mark, self.neighbor)
        # 系统自己会变棋盘， 所以还是要拿出来的
        chessboard[new_pos[0], new_pos[1]] = COLOR_NONE
        # 最后做出决策后更新信息
        # =============Find new pos(new pos is a list that contain (x, y)===================================
        # if the position of decision is not empty, return error
        assert chessboard[new_pos[0], new_pos[1]] == COLOR_NONE
        # add decision into candidate_list records the chess board
        self.candidate_list.append(new_pos)

    # get the last decision
    def get_result(self):
        return self.candidate_list[-1]


# 下了棋后，更新评估表的分数，中心包括自己加周边4个, 要传入评估表
def update_mark(position, chessboard, mark, field, color_main):
    hori = np.arange(position[0] - 4, position[0] + 5, 1)
    vert = np.arange(position[1] - 4, position[1] + 5, 1)
    for h in hori:
        for v in vert:
            tempt = (h, v)
            if tempt in field and chessboard[tempt[0], tempt[1]] == 0:
                chessboard[tempt[0], tempt[1]] = color_main
                score = evalocal(len(chessboard), (h, v), chessboard, color_main)
                chessboard[tempt[0], tempt[1]] = -color_main
                score1 = evalocal(len(chessboard), (h, v), chessboard, -color_main)
                chessboard[tempt[0], tempt[1]] = 0
                # test 打印当下了某个点后，另外某个点的评分
                # if position[0] ==2 and position[1] ==1:
                #     if h == 3 and v == 1:
                #         print(score,' ',score1)

                if color_main == 1:
                    mark[tempt[0], tempt[1]] = 1.1 * score + score1  #后手防守
                else:
                    mark[tempt[0], tempt[1]] = 1.3 * score + score1  # 先手进攻
    mark[position[0], position[1]] = -999999  # 下过的点分数更新为-1，以后不会再用到

def print_neighbor_mark(mark_table,neighbor_list):
    for x in neighbor_list:
        print('pos=',x,'mark=',mark_table[x[0],x[1]])

# 每当自己或者对手下了棋，更新neighbor的个数（包括减去已下的位置和更新加入周边的neighbor），要传入neighbor表
def update_neighbor(position, neighbor_list, chessboard, field):
    hori = np.arange(position[0] - 3, position[0] + 4, 1)
    vert = np.arange(position[1] - 3, position[1] + 4, 1)
    for h in hori:
        for v in vert:
            tempt = (h, v)
            if tempt in field:
                if chessboard[tempt[0], tempt[1]] == 0 and tempt not in neighbor_list:
                    neighbor_list.append(tempt)
    for e in neighbor_list:
        if chessboard[e[0], e[1]] != 0:
            neighbor_list.remove(e)


# 当前棋局评估总分(我的分数减去对面的分数), color是我
def evaluator_total(size, chessboard, color):
    return evaluator(size, chessboard, color) - evaluator(size, chessboard, -color)


# 评估棋盘某个颜色总分数, 墙也算对手
def evaluator(size, chessboard, color):
    total_score = 0
    total_score = 0
    live_three = 0
    dead_four = 0
    # 对行和列进行评估
    row_sp = np.vsplit(chessboard, size)
    col_sp = np.hsplit(chessboard, size)
    for i in range(size):
        row = [-color]
        col = [-color]
        for j in range(size):
            row.append(row_sp[i][0][j])
            col.append(col_sp[i][j][0])
        a1, b1, c1 = one_row(col, color)
        a2, b2, c2 = one_row(row, color)
        total_score = total_score + a1 + a2
        live_three = live_three + b1 + b2
        dead_four = dead_four + c1 + c2
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
        else:
            a1, b1, c1 = one_row(left1_incline, color)
            a2, b2, c2 = one_row(left2_incline, color)
            a3, b3, c3 = one_row(right1_incline, color)
            a4, b4, c4 = one_row(right2_incline, color)
            total_score = total_score + a1 + a2 + a3 + a4
            live_three = live_three + b1 + b2 + b3 + b4
            dead_four = dead_four + c1 + c2 + c3 + c4
        # 双三那几种情况分开来算
    if live_three + dead_four >= 2 :
        total_score += 11000
    else:
        total_score += live_three * 1100
        total_score += dead_four * 1100
    return total_score


# 启发估值，只估值当前落子中心向外4个位置的值，返回一个值
def evalocal(size, position, chessboard, color):
    x, y = position
    total_score = 0
    row = []
    col = []
    left_inc = []
    right_inc = []
    three = 0
    four = 0
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
            if x+i == -1  or y+i == -1:
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
    a1, b1, c1 = one_row(col, color)
    a2, b2, c2 = one_row(left_inc, color)
    a3, b3, c3 = one_row(right_inc, color)
    total_score = total_score+a+a1+a2+a3
    three = b+b1+b2+b3
    four = c+c1+c2+c3
    if three+four >= 2:
        total_score += 12000
    else:
        total_score += ((three+four) * 1100)
    return max(total_score, 8)

# 根据得分表对neighbor从小到达排序
def sort_neighbor(mark_table, neightbor_list):
    neightbor_list.sort(key=lambda x: mark_table[x[0], x[1]])

# Alpha-Beta 函数搜索, alpha一开始负无穷， beta一开始正无穷， 同时返回一个位置回来, 一开始给一个敌方的值
def alphabeta(point, neighbor, chessboard, mark_table, depth, maxmin_player, alpha, beta, field, color):
    return_pos = ()  # 待返回的东西
    # 启发分数到达一定，比如赢了那么大的分数，或者深度到了，就返回
    if depth == 0 or mark_table[point[0], point[1]] > 12000:
        return_pos = (point[0], point[1])
        return evaluator_total(len(chessboard), chessboard, color), return_pos

    if maxmin_player:
        best_value = float("-inf")  # 无穷小
        # 按照启发函数从大到小排出前15的点
        sort_neighbor(mark_table, neighbor)
        neighbor.reverse()
        if len(neighbor) > 15:
            neighbor = neighbor[0:15]  # 只取前15个
        print(neighbor)
        for pos in neighbor:
            # 深拷贝， 不影响原来数据，因为大家都要用
            new_neighbor = copy.deepcopy(neighbor)
            new_chessboard = copy.deepcopy(chessboard)
            new_mark_table = copy.deepcopy(mark_table)
            new_chessboard[pos[0], pos[1]] = color  # 假设下棋
            # 假设要下这个位置，更新下棋盘，评分表， neighbor，然后继续递归
            update_mark(pos, new_chessboard, new_mark_table, field, color)
            update_neighbor(pos, new_neighbor, new_chessboard, field)
            v, best_pos = alphabeta(pos, new_neighbor, new_chessboard, new_mark_table, depth-1, False, alpha, beta, field, color)
            if best_value < v:
                best_value = max(best_value, v)
                return_pos = pos  # 当前这个点可能是最好的选择
            alpha = max(alpha, best_value)
            if beta <= alpha:
                break
        return best_value, return_pos
    else:  # 考虑对手层
        best_value = float("inf")
        # 无穷大
        # 按照启发函数从小到大排出前15的点
        sort_neighbor(mark_table, neighbor)
        if len(neighbor) > 15:
            neighbor = neighbor[0:15]  # 只取前11个
        for pos in neighbor:
            # 深拷贝， 不影响原来数据，因为大家都要用
            new_neighbor = copy.deepcopy(neighbor)
            new_chessboard = copy.deepcopy(chessboard)
            new_mark_table = copy.deepcopy(mark_table)
            new_chessboard[pos[0], pos[1]] = -color  # 假设这个棋下到这了，是下对方的棋
            # 假设要下这个位置，更新下棋盘，评分表， neighbor，然后继续递归, 还是以我方的角度来看分数
            update_mark(pos, new_chessboard, new_mark_table, field, color)
            update_neighbor(pos, new_neighbor, new_chessboard, field)
            v, best_pos = alphabeta(pos, new_neighbor, new_chessboard, new_mark_table, depth - 1, True, alpha, beta,field,color)
            if best_value > v:
                best_value = min(best_value, v)
                return_pos = pos  # 目前最小的点
            beta = min(beta, best_value)
            if beta <= alpha:
                break
        return best_value, return_pos


# 让估值棋形换成黑棋估值
def neg(a):
    return -a


# 抽离出来的，对一行进行评估, i代表棋形的index, row是待评估的行，返回一个总分
def one_row(row, color):
    total_score = 0
    live_three = 0  # 理论来说同一行最多一个活三？
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
            time = KMP(row, cas)
            if case == 0:
                total_score += time * score
            if case == 1:
                live_three += time
            elif case == 2:
                death_four += time
            if live_three + death_four > 0: # 有可能出现同一行的冲四被当作活三的情况，其实一行最多一个冲四活三
                live_three = 1
                death_four = 0
    return (total_score, min(live_three, 1), min(death_four, 1))


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


