from asyncio import trsock
from distutils.log import error
from email.errors import NonPrintableDefect
import time
import math
import sys
import numpy as np 
import random
import ast

rate = .1

w1 = np.array([[1, -.5], [1, .5]])
w2 = np.array([[1, 2], [-1, -2]])
weights = [None,w1,w2]

b1 = np.array([[1, -1]])
b2 = np.array([[-.5,.5]])
biases = [None,b1,b2]

network = [weights,biases]

x1 = np.array([[2,3]])
y1 = np.array([[.8,1]])

inputs = [[x1],[y1]]

def sigmoid(num):
    return ( 1 / (1 + math.exp(-num)))

def dsig(num):
    val = num
    return val * (1 - val)

def errorcalc(A,layer,input):
    x,y = input
    a = x
    Av = np.vectorize(A)
    for j in range(1,len(layer)+1): 
        weightvectors,biasvectors = layer[0][j],layer[1][j]
        a.append(Av(a[j-1]@weightvectors+biasvectors))
    error = 0
    for i in range (0,len(y[0][0])):
        error = math.pow(y[0][0][i]-a[-1][0][i],2) + error
    return error/2

def backprop(A,layer,input):
    epochs = 5000
    Av = np.vectorize(A)
    for z in range(0,epochs):
        xs,ys = input 
        print("Epoch: " + str(z+1))
        for i in range(0,len(xs)):
            x,y = xs[i],ys[i]
            a = [x]
            sigs = [None] 
            for j in range(1,len(layer)+1): 
                weightvectors,biasvectors = layer[0][j],layer[1][j]
                sigs.append((Av(a[j-1]@weightvectors+biasvectors))*(1-Av(a[j-1]@weightvectors+biasvectors)))
                a.append(Av(a[j-1]@weightvectors+biasvectors))
            finaldelta = sigs[-1] * (y-a[-1])
            deltas = [finaldelta]
            for j in range (len(layer)-1,0,-1):
                weightvectors,biasvectors = layer[0][j+1],layer[1][j+1]
                deltas.append(np.multiply(np.array(sigs[j]),np.dot(deltas[-1],weightvectors.transpose())))
            newbias = [None]
            newweight = [None]
            for j in range(1,len(layer[0])):
                weightvectors,biasvectors = layer[0][j],layer[1][j]
                newbias.append(biasvectors + rate * deltas[len(deltas)  - j]) # bias + learning rate * deltas
                newweight.append(weightvectors + rate * np.array(a[j-1]).transpose()@deltas[len(deltas) - j]) # weight learning rate * inputs * deltas
            print(a[-1])
            layer = [newweight,newbias]
    return layer 

def circlecheck(pt):
    x,y = pt[0],pt[1]
    if math.pow((math.pow(x,2) + math.pow(y,2)),.5) <= 1: 
        return 1 
    return 0

def round(num):
    if num >= .5:
        return 1
    return 0 

def ptcheck(A, layer, input):
    Av = np.vectorize(A)
    Rv = np.vectorize(round)
    xs,ys = input 
    pts = [] 
    for i in range(0,len(xs)):
        x,y = xs[i],ys[i]
        a = [x]
        sigs = [None] 
        for j in range(1,len(layer)+1): 
            weightvectors,biasvectors = layer[0][j],layer[1][j]
            sigs.append((Av(a[j-1]@weightvectors+biasvectors))*(1-Av(a[j-1]@weightvectors+biasvectors)))
            a.append(Av(a[j-1]@weightvectors+biasvectors))
        pts.append(Rv(a[-1]))
    count = 0 
    total = 0 
    for i in range (0,len(ys)):
        if ys[i][0] - pts[i][0] == 0 :
            count += 1 
        total += 1 
    return total-count, (total-count)/1000


def circleprop (A,layer,input,learningrate):
    epochs = 500
    Av = np.vectorize(A)
    for z in range(0,epochs):
        xs,ys = input 
        print("Epoch: " + str(z+1))
        for i in range(0,len(xs)):
            x,y = xs[i],ys[i]
            a = [x]
            sigs = [None] 
            for j in range(1,len(layer[0])): 
                weightvectors,biasvectors = layer[0][j],layer[1][j]
                sigs.append((Av(a[j-1]@weightvectors+biasvectors))*(1-Av(a[j-1]@weightvectors+biasvectors)))
                a.append(Av(a[j-1]@weightvectors+biasvectors))
            finaldelta = sigs[-1] * (y-a[-1])
            deltas = [None for k in range(len(layer[0]))]
            deltas[-1] = finaldelta
            for j in range (len(layer[0])-2,0,-1):
                weightvectors,biasvectors = layer[0][j+1],layer[1][j+1]
                deltas [j] = (sigs[j]*(deltas[j+1]@weightvectors.transpose()))
            newbias = [None]
            newweight = [None]
            for j in range(1,len(layer[0])):
                weightvectors,biasvectors = layer[0][j],layer[1][j]
                newbias.append(biasvectors + learningrate * deltas[j]) # bias + learning rate * deltas
                newweight.append(weightvectors + learningrate * np.array(a[j-1]).transpose()@deltas[j]) # weight learning rate * inputs * deltas
            layer = [newweight,newbias]
        count,learningrate = ptcheck(A,layer,input)
        print("pts classified incorrectly: " + str(count))
    return layer 


if sys.argv[1] == "S":

    ws = [None, 2 * np.random.rand(2, 2) - 1, 2 * np.random.rand(2, 2) - 1]
    bs = [None, 2 * np.random.rand(1, 2) - 1, 2 * np.random.rand(1, 2) - 1]

    xinput = [np.array([[0, 0]]), np.array([[0, 1]]), np.array([[1, 0]]), np.array([[1, 1]])]
    yinput = [np.array([[0, 0]]), np.array([[0, 1]]), np.array([[0, 1]]), np.array([[1, 0]])]

    backprop(sigmoid,[ws,bs],[xinput,yinput])

elif sys.argv[1] == "C":
    x_list = [np.array(2*np.random.rand(1,2)-1)for k in range(10000)]
    y_list = [np.array([[circlecheck(coords[0])]])for coords in x_list]

    # ws = [None, 2 * np.random.rand(2, 4) - 1, 2 * np.random.rand(4, 1) - 1]
    # bs = [None, 2 * np.random.rand(1, 4) - 1, 2 * np.random.rand(1, 1) - 1]
    ws = [None, 2 * np.random.rand(2, 12) - 1, 2 * np.random.rand(12, 4) - 1, 2 * np.random.rand(4, 1) - 1]
    bs = [None, 2 * np.random.rand(1, 12) - 1, 2 * np.random.rand(1, 4) - 1, 2 * np.random.rand(1, 1) - 1]
 


    rate = .5 

    circleprop(sigmoid,[ws,bs],[x_list,y_list],rate)

    print(None)

else:
    print("input error")


