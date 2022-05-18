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

    def aStarAlgo(start_node, stop_node):         
            open_set = set(start_node) 
            closed_set = set()
            g = {}  #store distance from starting node
            parents = {}  #contains an adjacency map of all nodes        
            g[start_node] = 0
            parents[start_node] = start_node        
            while len(open_set) > 0:
                n = None  #node with lowest f() is found
                for v in open_set:
                    if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):
                        n = v            
                if n == stop_node or Graph_nodes[n] == None:
                    pass
                else:
                    for (m, weight) in get_neighbors(n):
                        if m not in open_set and m not in closed_set:
                            open_set.add(m)
                            parents[m] = n
                            g[m] = g[n] + weight                      
                        else:
                            if g[m] > g[n] + weight:
                                g[m] = g[n] + weight
                                parents[m] = n
                                if m in closed_set:
                                    closed_set.remove(m)
                                    open_set.add(m)
                if n == None:
                    print('Path does not exist!')
                    return None
                if n == stop_node:
                    path = []
                    while parents[n] != n:
                        path.append(n)
                        n = parents[n]
                    path.append(start_node)
                    path.reverse()
                    print('Path found: {}'.format(path))
                    return path
                open_set.remove(n)
                closed_set.add(n)
            print('Path does not exist!')
            return None

    #function to return neighbor and its distance from the passed node
    def get_neighbors(v):
        if v in Graph_nodes:
            return Graph_nodes[v]
        else:
            return None

    #for simplicity we consider heuristic distances given and this function returns heuristic distance for all nodes
    def heuristic(n):
            H_dist = {
                'A': 0,
                'B': 7,
                'C': 2,
                'D': 5,
                'E': 2,             
            }
            return H_dist[n]

    #Describe your graph here  
    Graph_nodes = {
        'B': [('A', 7), ('C', 6), ('D', 3), ('E', 3)],
        'A': [('B', 7), ('E', 2)],
        'C': [('B', 6), ('D', 6)],
        'D': [('B', 3), ('C', 6), ('E', 2)],
        'E': [('A', 2), ('B', 3), ('D', 2)]}
    aStarAlgo('A', 'C')
