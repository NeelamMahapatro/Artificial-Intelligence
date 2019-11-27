# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:38:18 2019

@author: student
"""

import numpy as np
import sys

maxv=9999999

def priorm(m):
    c=0
    for i in range(3):
        for j in range(3):
            if m[i][j]!=0:
                if m[i][j]!=i*3+j+1:
                    c+=1
    return c

m=[[1,2,3],
   [0,4,6],
   [7,5,8]]
m=np.array(m)

print('Initial state :')
print(m)

def move(g,m):
    for i in range(3):
        for j in range(3):
            if m[i][j]==0:
                k=i
                l=j
    #print(k)
    #print(l)
    fl=[]
    t=[]
    h=priorm(m)
    if h!=0:
        if k>0:
            t1=m.copy()
            t1[k][l]=t1[k-1][l]
            t1[k-1][l]=0
            ft=priorm(t1)
            #print(ft)
            #print(t1)
            fl.append(ft)
            t.append(t1)
        else:
            fl.append(maxv)
            t.append(0)
        if k<2:
            t2=m.copy()
            t2[k][l]=t2[k+1][l]
            t2[k+1][l]=0
            ft=priorm(t2)
            #print(ft)
            #print(t2)
            fl.append(ft)
            t.append(t2)
        else:
            fl.append(maxv)
            t.append(0)
        if l>0:
            t3=m.copy()
            t3[k][l]=t3[k][l-1]
            t3[k][l-1]=0
            ft=priorm(t3)
            #print(ft)
            #print(t3)
            fl.append(ft)
            t.append(t3)
        else:
            fl.append(maxv)
            t.append(0)
        if l<2:
            t4=m.copy()
            t4[k][l]=t4[k][l+1]
            t4[k][l+1]=0
            ft=priorm(t4)
            #print(ft)
            #print(t4)
            fl.append(ft)
            t.append(t4)
        else:
            fl.append(maxv)
            t.append(0)
        fl=np.array(fl)
        minf=min(fl)
        c=0
        for i in range(4):
            if fl[i]==minf:
                c=i
        n=t[c]
        print('g='+str(g+1)+', h='+str(minf)+', f=g+h='+str(g+1+minf))
        print(n)
        move(g+1,n)
        
move(0,m)  