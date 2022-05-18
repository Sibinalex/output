# output


=======================================================

#DFS

    graph={1: [2, 3, 4], 2: [1, 3], 3: [4], 4: [1, 3, 4]}
    
    visited = []
    
    def dfs(visited,graph,root):
      if root not in visited:
        print(root)
        visited.append(root)
        for neighbour in graph[root]:
          dfs(visited,graph,neighbour)
    dfs(visited,graph,1)




=======================================================

#BFS

    visited = [] 
    queue = []     
    graph = {0:[1,2,3], 1:[0,2], 2:[0,1,4], 3:[0], 4:[2]}



    def bfs(visited, graph, node): 

      visited.append(node)
      queue.append(node)

      while queue:
        m = queue.pop(0) 
        print (m, end = " ") 

        for neighbour in graph[m]:
          if neighbour not in visited:
            visited.append(neighbour)
            queue.append(neighbour)


    print("Following is the Breadth-First Search")
    bfs(visited, graph, 0)

=======================================================

#Prims Algorithm

    INF = 9999999
    # number of vertices in graph
    N = 5
    #creating graph by adjacency matrix method

    G = [[0, 4, 0, 3, 5],[4, 0, 2, 0, 0], [0, 2, 0, 1, 0], [3, 0, 1, 0, 0], [5, 0, 0, 0, 0]]

    selected_node = [0, 0, 0, 0, 0] 
    no_edge = 0

    selected_node[0] = True

    # printing for edge and weight 
    print("Edge : Weight") 
    while (no_edge < N - 1):
        minimum = INF 
        a=0
        b=0
        for m in range(N):
            if selected_node[m]: 
                for n in range(N):
                    if ((not selected_node[n]) and G[m][n]): # not in selected and there is an edge
                        if minimum > G[m][n]:
                            minimum = G[m][n] 
                            a=m
                            b =n
        # print(str(a) + "-" + str(b) + ":" + str(G[a][b]))
        print(a, "-->" , b , ":" , G[a][b])
        selected_node[b] = True 
        no_edge += 1



=======================================================

#Graph Coloring

    def addEdge(adj, v, w):
        adj[v].append(w)
        adj[w].append(v)
        return adj
    def graph_coloring(adj, V):
        result = [-1] * V
        result[0] = 0
        avail = [False] * V
        for u in range(1, V):
            for i in adj[u]:
                if (result[i] != -1):
                    avail[result[i]] = True
                    cr = 0
                    while cr < V:
                        if (avail[cr] == False):
                            break
                        cr += 1
                        result[u] = cr

        for i in adj[u]:
            if (result[i] != -1):
                avail[result[i]] = False
            for u in range(V):
                print("Vertex", u, " ---> Color", result[u])

    n = int(input("Enter number of vertices : "))
    c = int(input("Enter number of colors : "))
    edges=int(input("Enter numberof edges"))
    gr=[[] for i in range(n)]
    print("Enter edges : ")
    for i in range(edges):
        u=int(input("u "))
        v=int(input("v "))
        addEdge(gr,u,v)
    print("Coloring of graph 1 ")
    graph_coloring(gr, n)

=========================================================

A star

    class Node:
        def __init__(self,data,level,fval):
            self.data = data
            self.level = level
            self.fval = fval
        def generate_child(self):
            x,y = self.find(self.data,'_')
            val_list = [[x,y-1],[x,y+1],[x-1,y],[x+1,y]]
            children = []
            for i in val_list:
                child = self.shuffle(self.data,x,y,i[0],i[1]) 
                if child is not None:
                    child_node = Node(child,self.level+1,0)
                    children.append(child_node) 
            return children

        def shuffle(self,puz,x1,y1,x2,y2):
            if x2 >= 0 and x2 < len(self.data) and y2 >= 0 and y2 < len(self.data):
                temp_puz = []
                temp_puz = self.copy(puz)
                temp = temp_puz[x2][y2] 
                temp_puz[x2][y2] = temp_puz[x1][y1] 
                temp_puz[x1][y1] = temp
                return temp_puz
            else:
                return None

        def copy(self,root):
            temp = []
            for i in root:
                t = []
                for j in i:
                    t.append(j) 
                temp.append(t)
            return temp
        def find(self,puz,x):
            for i in range(0,len(self.data)):
                for j in range(0,len(self.data)): 
                    if puz[i][j] == x:
                        return i,j

    class Puzzle:
        def __init__(self,size):
            self.n = size
            self.open = []
            self.closed = []
        def accept(self):
            puz = []
            for i in range(0,self.n):
                temp = input().split(" ")
                puz.append(temp) 
            return puz

        def f(self,start,goal):
            return self.h(start.data,goal)+start.level

        def h(self,start,goal):
            temp = 0
            for i in range(0,self.n):
                for j in range(0,self.n):
                    if start[i][j] != goal[i][j] and start[i][j] != '_':
                        temp += 1 
            return temp

        def process(self):
            print("Enter the start state matrix \n") 
            start = self.accept()
            print("Enter the goal state matrix \n") 
            goal = self.accept()
            start = Node(start,0,0)
            start.fval = self.f(start,goal)
            self.open.append(start)
            print("\n\n")
            while True:
                cur = self.open[0] 
                print("")
                print(" | ") 
                print(" | ") 
                print(" \\\'/ \n") 
                for i in cur.data:
                    for j in i: 
                        print(j,end=" ")
                    print("")
                if(self.h(cur.data,goal) == 0):
                    break
                for i in cur.generate_child():
                    i.fval = self.f(i,goal)
                    self.open.append(i)
                self.closed.append(cur)
                del self.open[0]
                self.open.sort(key = lambda x:x.fval,reverse=False)

    puz = Puzzle(3) 
    puz.process()
