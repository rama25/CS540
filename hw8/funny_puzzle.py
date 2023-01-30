import numpy as np
from collections import defaultdict
import heapq as hq


def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    # correct
    arr = np.reshape(from_state, (3, 3))
    expected = np.array([1, 2, 3, 4, 5, 6, 7, 0, 0])
    expected = np.reshape(expected, (3, 3))
    man_dis = 0
    for i in range(1, 8):
        expected_pos = np.argwhere(expected == i)[0]
        real_pos = np.argwhere(arr == i)[0]
        man_dis += (abs(expected_pos[0] - real_pos[0]) + abs(expected_pos[1] - real_pos[1]))
    return man_dis


def swap_pos(arr, r, c):
    swap_list = []
    if (r == 0 or r == 1) and arr[(r+1, c)]!=0:
        swap_list.append((r + 1, c))
    if (c == 0 or c == 1) and arr[(r, c+1)]!=0:
        swap_list.append((r, c + 1))
    if (r == 1 or r == 2) and arr[(r-1,c)]!=0:
        swap_list.append((r - 1, c))
    if (c == 1 or c == 2) and arr[(r, c-1)]!=0:
        swap_list.append((r, c - 1))
    return swap_list

def get_succ(state):
    temp_state = state
    successors = []
    temp_state = np.reshape(temp_state, (3, 3))
    zero_pos = np.argwhere(temp_state == 0)
    r2, c2 = zero_pos[1]
    r1, c1 = zero_pos[0]
    swap2 = swap_pos(temp_state, r2, c2)
    swap1 = swap_pos(temp_state, r1, c1)

    for move in swap2:
        copy_state = np.copy(temp_state)
        copy_state[move], copy_state[(r2, c2)] = copy_state[(r2, c2)], copy_state[move]
        flat_list = copy_state.flatten().tolist()
        successors.append((flat_list, get_manhattan_distance(flat_list)))

    for move in swap1:
        copy_state = np.copy(temp_state)
        copy_state[move], copy_state[(r1, c1)] = copy_state[(r1, c1)], copy_state[move]
        flat_list = copy_state.flatten().tolist()
        successors.append((flat_list, get_manhattan_distance(flat_list)))

    successors = sorted(successors)
    return successors

def print_succ(state):
    res = get_succ(state)
    for i, j in res:
        print(i, "h=" + str(j))
    return res

# Below function discussed with Deepak Ranganathan and Ojal Sethi
def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    node_dict = defaultdict(None)
    visited = set()

    temp_state = state
    g = 0
    h = get_manhattan_distance(temp_state)

    pq = []
    hq.heappush(pq, (g + h, temp_state, (g, h, -1, 0)))
    visited.add((tuple(temp_state)))
    iter = 0
    max_q_len = 1
    node_dict[iter] = (temp_state, -1, h)
    iter += 1
    ans = -1

    while len(pq):
        item = list(hq.heappop(pq))
        if item[2][1] == 0:
            ans = item
            break

        success_states = get_succ(item[1])
        for state, h in success_states:
            if tuple(state) not in visited:
                visited.add(tuple(state))
                g = item[2][0] + 1
                hq.heappush(pq, (g + h, state, (g, h, item[2][3], iter)))
                node_dict[iter] = (state, item[2][3], h)
                iter += 1
            max_q_len = max(max_q_len, len(pq))

    current = ans[2][3]
    result = []

    while current != -1:
        result.append(node_dict[current])
        current = node_dict[current][1]

    for j, i in enumerate(result[::-1]):
        print(i[0], "h=" + str(i[2]) + " moves: " + str(j))

    print(f"Max queue length: {max_q_len}")


if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """

    # print_succ([3, 4, 6, 0, 0, 1, 7, 2, 5])

    # print_succ([2,5,1,4,0,6,7,0,3])
    # print()
    #
    # print(manhattan_dist([2,5,1,4,0,6,7,0,3]))
    # print()

    
    solve([4,3,0,5,1,6,7,2,0])
    # print()

    # print_succ([6, 0, 0, 3, 5, 1, 7, 2, 4])
    # print()

    # print_succ([2,5,1,4,3,6,7,0,0])
    # print()

    # print(succe([3, 4, 6, 0, 0, 1, 7, 2, 5]))
    #
    # print(succe([3, 0, 6, 0, 4, 1, 7, 2, 5]))
