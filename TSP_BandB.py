inf = 1000
n = 5
inp = [[inf, 1, 6, 8, 4],
         [7, inf, 8, 5, 6],
         [6, 8, inf, 9, 7],
         [8, 5, 9, inf, 8],
         [4, 6, 7, 8, inf]]


class Node:

    def __init__(self, cost_, graph_, vis_, id_):
        self.cost = cost_
        self.graph = graph_
        self.vis = vis_
        self.id = id_

def reduce(lst):
    cost = 0
    global n, inf

    for i in range(n):
        num = inf
        for j in range(n):
            if lst[i][j] != inf:
                num = min(num, lst[i][j])
        if num != inf:
            for j in range(n):
                if lst[i][j] != inf:
                    lst[i][j] = lst[i][j]-num
        if num != inf:
            cost = cost + num
    for i in range(n):
        num = inf
        for j in range(n):
            if lst[j][i] != inf:
                num = min(num, lst[j][i])
        if num != inf:
            for j in range(n):
                if lst[j][i] != inf:
                    lst[j][i] = lst[j][i]-num
        if num != inf:
            cost = cost + num
    return cost

bound = inf

def copy_arr(arr):
    copy = []
    for i in range(len(arr)):
        x = []
        for j in range(len(arr[i])):
            x.append(arr[i][j])
        copy.append(x)
    return copy

def copy_vis(arr):
    lst = []
    for i in range(len(arr)):
        lst.append(arr[i])
    return lst

compute = []
temp = copy_arr(inp)
c1 = reduce(temp)
compute.append(Node(c1, temp, [0], 0))

while len(compute) > 0:
    Temp = compute.pop()                                                                                     
    if bound <= Temp.cost:
        continue
    for i in range(n):
        if Temp.graph[Temp.id][i] != inf and i not in Temp.vis:
            new_list = copy_arr(Temp.graph)
            new_id = i
            new_vis = copy_vis(Temp.vis)
            new_vis.append(i)
            new_cost = Temp.cost + Temp.graph[Temp.id][i] + reduce(new_list)
            newNode = Node(new_cost, new_list, new_vis, i)
            stack = []
            while len(compute) > 0 and compute[-1].cost > newNode.cost:
                stack.append(compute.pop())
            compute.append(newNode)
            while len(stack)>0:
                compute.append(stack.pop())
            if len(newNode.vis) == n:
                bound = min(bound, newNode.cost)

print(bound)