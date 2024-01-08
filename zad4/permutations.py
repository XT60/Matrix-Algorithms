import numpy as np 
import networkx as nx
from collections import deque



class PermutationEngine:

    def __init__(self, algorithm):
        self.permutation_algorithm = {'minimum_degree': self._minimum_degree_permutation,
                            'cuthill_mckee': self._cuthill_mckee,
                            'reversed_cuthill_mckee': self._reversed_cuthill_mckee}.get(algorithm)


    def _minimum_degree_permutation(self, matrix):
        n,m = matrix.shape
        ADJ = {i:set() for i in range(n)}
        for i in range(n):
            for j in range(m):
                if matrix[i][j] != 0 and i != j:
                    ADJ[i].add(j)
        permutation = []
        for i in range(n):
            deg_min = m+1
            for v, adj in ADJ.items():
                if len(adj) < deg_min:
                    v_min = v
                    deg_min = len(adj)
            for v in ADJ:
                ADJ[v] = ADJ[v].difference([v_min])
            for u in ADJ[v_min]:
                ADJ[u] = (ADJ[u].union(ADJ[v_min].difference([u])))
            ADJ.pop(v_min)
            permutation.append(v_min)
        return permutation

    def _cuthill_mckee(self, matrix):

        def BFS():
            while Q:
                v = Q.popleft()
                if Visited[v]:
                    continue
                permutation.append(v)
                Visited[v] = True
                for u in sorted(nx.neighbors(G, v), key = lambda x : G.degree(x)):
                    if not Visited[u]:
                        Q.append(u)
        n = len(matrix)
        
        G = nx.Graph()
        G.add_nodes_from(range(n))
        
        for i in range(n):
            for j in range(n):
                if matrix[i][j] != 0 and i != j:
                    G.add_edge(i, j)

        permutation = []
        Visited = [False for i in range(n)]
        sorted_nodes = sorted([x for x in G.degree()], key = lambda x : x[1])
        # print(sorted_nodes)
        sorted_nodes = list(map(lambda x : x[0], sorted_nodes))
        Q = deque()
        for s in sorted_nodes:
            if not Visited[s]:
                Q.append(s)
                BFS()
        
        return permutation

    def _reversed_cuthill_mckee(self, matrix):
        return self._cuthill_mckee(matrix)[::-1]

    def permutate(self, matrix):
        permutation = self.permutation_algorithm(matrix)
        new_matrix = matrix.copy()
        for i in range(len(permutation)):
            if i == permutation[i]:
                continue
            new_matrix[i,:] = matrix[permutation[i],:].copy()

        matrix = new_matrix.copy()
        for i in range(len(permutation)):
            if i == permutation[i]:
                continue
            new_matrix[:,i] = matrix[:,permutation[i]].copy()

        return new_matrix