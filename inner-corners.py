#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:07:52 2022

@author: jemmaschroder
"""

import numpy as np
import matplotlib.pyplot as plt
import time
start_time = time.time()



#------- INITALIZE EVERYTHING -------


colors =['tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:orange', 
         'tab:cyan', 'tab:olive', 'tab:pink', 'tab:brown']

# b is the value of P(1,b)
#n is number it iterations you want
# mathplotlib is annoying and normalizing tick marks is annoying so photo is not neccessarily to scale

n=2
b=np.sqrt(3)
polygon_full=np.empty([n,5,2])

#initating points of polygon
origin=np.array([0,0])
p1=np.array([b,0])
p2=np.array([b,1])
p3=np.array([0,1])

#okay so basically the way i'm storying data is by storing it as a 2x3xn tensor (numpy array)
#where the nth slice is the points of the nth mutation
polygon_full[0]=np.array([origin,p1,p2,p3,origin])

#and we're going to do the same thing to store the transformation matrices
#i dont actually know if we need to store these?? but still maybe good to do
#(i think it's faster because i reference it multiple times so don't have to redo computation)
transformation_matrices=np.empty([n,2,2])

#initiate eigendirections (can hardcode these bc constant for polydisks)

ed1=np.array([-1,1]) #based at p1
ed2=np.array([-1,-1]) #based at p2
ed3=np.array([1,-1]) #based at p3

#put all points in tensor, same deal as above
eigendirections=np.empty([n,3,2])
eigendirections[0]=np.array([ed1,ed2,ed3])

#------- END INITIALIZAZTION -------

#checks whether points are counterclockwise

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


#find intersection point
def find_intersection(shape,eds,vertex: int):
    pointlist=[]
    j=0
    for i in range(0,4):
        A=shape[i]
        B=shape[i+1]
        C=shape[vertex]
        D=shape[vertex]+eds[vertex-1]
        
        #check whether it intersects
        if intersect(A,B,C,D)==True:
            #compute intersection point
            #check if intersects on horizontal
            j+=1
            #--- WIP FOR CASE WHERE LAST INTERSECTION IS ON CORNER OR NOT CORRECT ONE
            if A[0]==B[0]:
                #compute actual intersection point
                pointlist.append(np.array([A[0],(C[1]-D[1])/(C[0]-D[0])*(A[0]-C[0])+C[1]]))
                if np.array_equal(pointlist(-1),A) or np.array_equal(pointlist(-1),B):
                    pointlist.pop()
                    # print(f'The line from {shape[i]} to {shape[i+1]} and the line from {eds[vertex]} to {shape[vertex+1]} do not intersect')
                # else:
                     # print(f'The line from {shape[i]} to {shape[i+1]} and the line from {eds[vertex]} to {shape[vertex+1]} intersect at {point}')
                    
            #otherwise it intersects on vertical
            else:
                point=np.array([(C[0]-D[0])/(C[1]-D[1])*(A[1]-C[1])+C[0],A[1]])
                if np.array_equal(point,A) or np.array_equal(point,B):
                   point=None
            
    return (point)

#plotting for polygon
#where n is the nth iteration you want to plot
def plot_polygon(tensor, n):
    shape_list = []
    for i in range(0,len(tensor[n])):
        shape_list.append(tensor[n][i])
    xs, ys = zip(*shape_list) #create lists of x and y values
    plt.plot(xs,ys) 
    
#plotting for nodal rays
def plot_eigendirection(ev_tensor, polygon_tensor,n):
    for i in range (0,3):
        ev=[polygon_tensor[n][i+1],polygon_tensor[n][i+1]+ev_tensor[n][i]]
        xe, ye = zip(*ev) #create lists of x and y values
        plt.plot(xe,ye, color=colors[i])
    
    

#------- CODE FOR TRANSFORMATION -------

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm


#preserve is vector you want preserved, change is vector you want changed, change_to is new vector
def transformation(preserve,change, change_to):
    vec = np.concatenate((preserve,change_to))
    mat = np.array([ [preserve[0], preserve[1], 0,   0  ],
              [0,   0,   preserve[0], preserve[1]], 
              [change[0], change[1], 0,  0,  ],
              [0,   0,   change[0], change[1]], ])
    res = np.linalg.solve(mat,vec)
    return np.array([ [res[0],res[1]], 
              [res[2],res[3]] ])

transformation_matrices[0] = transformation(ed1,p2-p1,normalize(p1))

#given an nodal ray, compute transformation matrix
#where p is from 1 to 3 and represents the point you want to transform around
def transformation_from_ev(p):
    transformation
    return

# transformation_matrices = np.vstack([transformation_matrices, transformation(ed1,p2-p1,normalize(p1))])
transformation_matrices[1] = transformation(ed1,p2-p1,normalize(p1))


print(f'transformation: {transformation(ed1,p2-p1,normalize(p1))}')

point = find_intersection(polygon_full[0],eigendirections[0],1)

plt.figure()
plot_polygon(polygon_full,0)
plot_eigendirection(eigendirections,polygon_full,0)


#plot intersection point
plt.plot(point[0],point[1],color='black', marker='o')



polygon_full[1]=np.array([origin, np.matmul(transformation_matrices[0], p2-p1)+p1, point, p3, origin])
plot_polygon(polygon_full,1)

for i in range(0,3):
    eigendirections[1][i]=np.matmul(transformation_matrices[0],eigendirections[0][i])+polygon_full[1][i+1]
    
    
print(eigendirections[1])

plot_eigendirection(eigendirections,polygon_full,1)


plt.show() 
    
print("--- %s seconds ---" % (time.time() - start_time))

        
    
    
    
    

    



               
               