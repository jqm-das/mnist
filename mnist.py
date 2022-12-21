from asyncio import trsock
from distutils.log import error
from email.errors import NonPrintableDefect
import re
import time
import math
import sys
import numpy as np 
import random
import ast
import pickle
import csv

file_train = "mnist_train.csv"
file_test = "mnist_test.csv"

def sigmoid(num):
    return ( 1 / (1 + math.exp(-num)))

def read_train(f):
    keys = dict()
    for i in range (0,10):
        list = [0 for i in range (0,10)]
        list[i] = 1
        keys[i] = list
    xs = []
    ys = [] 
    with open(f) as file:
        reader = csv.reader(file,delimiter=",")
        for line in reader:
            ys.append(np.array(keys.get(int(line[0]))))
            x = [] 
            for i in range(1,len(line)):
                x.append(float(line[i])/255)
            xs.append(np.array([x]))
    return xs,ys

def network_create(perceptrons):
    weights = [None]
    biases = [None]
    for i in range (0,len(perceptrons)-1):
        weights.append(2*np.random.rand(perceptrons[i],perceptrons[i+1]) - 1)
        biases.append(2*np.random.rand(1,perceptrons[i+1])-1)
    return weights,biases

def find_greatest(ls):
    ind = 0 
    ls = ls[0]
    for i in range(1,len(ls)):
        if ls[i] > ls[ind]:
            ind = i
    for i in range(0,len(ls)):
        if i == ind:
            ls[i] = 1
        else:
            ls[i] = 0 
    return ls 

def ptcheck(A, layer, input):
    Av = np.vectorize(A)
    xs,ys = input 
    pts = [] 
    for i in range(0,len(xs)):
        x,y = xs[i],ys[i]
        a = [x]
        sigs = [None] 
        for j in range(1,len(layer[0])): 
            weightvectors,biasvectors = layer[0][j],layer[1][j]
            sigs.append((Av(a[j-1]@weightvectors+biasvectors))*(1-Av(a[j-1]@weightvectors+biasvectors)))
            a.append(Av(a[j-1]@weightvectors+biasvectors))
        pts.append(find_greatest(a[-1]))
    count = 0 
    total = 0 
    for i in range (0,len(ys)):
        counter = 0 
        for j in range (0,len(ys[i])):
            if ys[i][j] == pts[i][j]:
                counter += 1 
        if counter == 10:
            count += 1 
        total += 1 
    print("Percent inaccuracy: %s" % (100* ((total-count)/total)) + "%")
    return total-count, (total-count)/10000

def backprop (A,layer,input,learningrate):
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
        savedweights = open("previous_weights","wb")
        pickle.dump(newweight,savedweights)
        savedweights.close()
        savedbiases = open("previous_biases","wb")  
        pickle.dump(newbias,savedbiases)
        savedbiases.close()
        print("pts classified incorrectly: " + str(count))
    return layer 

def trainingdata():
    x_ls,y_ls = read_train(file_train)

    x_train = open("x_train","wb")
    pickle.dump(x_ls,x_train)
    x_train.close()

    y_train = open("y_train","wb")
    pickle.dump(y_ls,y_train)
    y_train.close()

    return x_ls,y_ls 


def testdata():
    x_ls,y_ls = read_train(file_test)

    x_test = open("x_test","wb")
    pickle.dump(x_ls,x_test)
    x_test.close()

    y_test = open("y_test","wb")
    pickle.dump(y_ls,y_test)
    y_test.close()
    
    return x_ls,y_ls

def readtrain():
    x_train = open("x_train","rb")
    x_ls = pickle.load(x_train)
    x_train.close()

    y_train = open("y_train","rb")
    y_ls = pickle.load(y_train)
    y_train.close()

    return x_ls,y_ls

def readtest():
    x_test = open("x_test","rb")
    x_ls = pickle.load(x_test)
    x_test.close()

    y_test = open("y_test","rb")
    y_ls = pickle.load(y_test)
    y_test.close()

    return x_ls,y_ls

def readweights():
    w_file = open("previous_weights","rb")
    w_saved = pickle.load(w_file)
    w_file.close()

    b_file = open("previous_biases","rb")
    b_saved = pickle.load(b_file)
    b_file.close()

    return w_saved,b_saved


# 30 epochs completed

if sys.argv[1] == "S":
    x_training,y_training = trainingdata()
    x_testing,y_testing = testdata()
    w_current,b_current = network_create([784,300,100,10])
else:
    x_training,y_training = readtrain()
    x_testing,y_testing = readtest()
    w_current,b_current = readweights()


newlayer = ptcheck(sigmoid,[w_current,b_current],[x_testing,y_testing])
# print(ptcheck(sigmoid,[w_ls,b_ls],[x_ls,y_ls]))


