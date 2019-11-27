# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 09:59:35 2019

@author: student
"""
import sys
import math
import itertools


def tsp(graph, s):
    lis = []
    for i in range(0, len(graph[0])):
        if i != s:
            lis.append(i)
    min_path = sys.maxsize
    perm = list(itertools.permutations(lis))
    l = len(perm) - 1
    while l >= 0:
        curr_path_weight = 0
        k = s
        lis = perm[l]
        for i in range(0, len(lis)):
            curr_path_weight = curr_path_weight + graph[k][lis[i]]
            k = lis[i]
        curr_path_weight += graph[k][s]
        min_path = min(min_path, curr_path_weight)
        l = l - 1
        
        
    return min_path


graph = [[0, 1, 6, 8, 4], 
         [7, 0, 8, 5, 6], 
         [6, 8, 0, 9, 7], 
         [8, 5, 9, 0, 8], 
         [4, 6, 7, 8, 0]]
print(tsp(graph, 0))
    
    
    
        

