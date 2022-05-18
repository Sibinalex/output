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
