import numpy as np 
import networkx as nx
from collections import deque



class PermutationEngine:

    def __init__(self, algorithm):
        self.permutation_algorithm = {'minimum_degree': self._minimum_degree_permutation,
                            'cuthill_mckee': self._cuthill_mckee,
                            'reversed_cuthill_mckee': self._reversed_cuthill_mckee}.get(algorithm)


    def _minimum_degree_permutation(self, matrix):
        # Initialisation
        n = len(matrix)
        permutation = []
        G = {i:set() for i in range(n)}

        for i in range(n):
            for j in range(n):
                if matrix[i][j] != 0 and i != j:
                    G[i].add(j)
        
        # Algorithm
        for i in range(n):
            min_degree = n+1
            for v, adj in G.items():
                if len(adj) < min_degree:
                    p = v
                    min_degree = len(adj)
            for v in G:
                G[v] = G[v].difference([p])
            for u in G[p]:
                G[u] = (G[u].union(G[p].difference([u])))
            G.pop(p)
            permutation.append(p)


        return permutation

    def _cuthill_mckee(self, matrix):         
        # Initialisation

        n = len(matrix)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        
        for i in range(n):
            for j in range(n):
                if matrix[i][j] != 0 and i != j:
                    G.add_edge(i, j)

        permutation = []
        visited = [False for i in range(n)]
        sorted_nodes = sorted([x for x in G.degree()], key = lambda x : x[1])

        sorted_nodes = list(map(lambda x : x[0], sorted_nodes))
        Q = deque()

        # BFS Algorithm
        for s in sorted_nodes:
            if not visited[s]:
                Q.append(s)
                while Q:
                    v = Q.popleft()
                    if not visited[v]:
                        permutation.append(v)
                        visited[v] = True
                        for u in sorted(nx.neighbors(G, v), key = lambda x : G.degree(x)):
                            if not visited[u]:
                                Q.append(u)
        
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