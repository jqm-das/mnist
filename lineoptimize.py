import math
import sys


start = (0,0)

def A(x,y):
    return (4 * math.pow(x,2)) - (3 * x * y) + (2 * math.pow(y,2)) + (24 * x) - (20 * y)

def B(x,y):
    return math.pow((1-y),2) + math.pow((x-math.pow(y,2)),2)

def partialAx(x,y):
    return (8 * x) - (3 * y) + 24 

def partialAy(x,y):
    return (4*y) - (3 * x) - 20 

def partialBx(x,y):
    return (2 * (x - math.pow(y,2)))

def partialBy(x,y):
    return (2 * (y - 1 )) - (4 * y * (x - math.pow(y,2)))

def magnitude(pt):
    x,y = pt 
    return math.sqrt(math.pow(x,2)+ math.pow(y,2))


def exfunction(x):
    return math.sin(x) + math.sin(3*x) + math.sin(4*x)

def make_funct(a,b):
    def funct(x):
        return a*x + b
    return funct 

f = make_funct(2,3)
g = make_funct(4,5)

# print(f(7),g(7))

def make_grad(f,x,y,partx,party):
    def grad(learningrate):
        return f(x-learningrate*partx,y-learningrate*party)
    return grad 

def one_d_minimize(f,left,right,tolerance):

    if right - left < tolerance:
        return (right+left)/2

    interval = right - left
    onethird = interval * (1/3) + left
    twothird = interval * (2/3) + left
    
    if f(twothird) > f(onethird):
        return one_d_minimize(f,left,twothird,tolerance)
    else:
        return one_d_minimize(f,onethird,right,tolerance)
 
rate = .1

if sys.argv[1] == "A":
    delta = (partialAx(0,0),partialAy(0,0))
    x,y = start 
    while magnitude(delta) > math.pow(10,-8) or -1 * magnitude(delta) > math.pow(10,-8):
        dx,dy = delta 
        grad = make_grad(A,x,y,dx,dy)#Line optimize
        rate = one_d_minimize(grad,0,1,math.pow(10,-10))#Line optimize
        print("learning rate: " + str(rate))
        x = x - rate * dx
        y = y - rate * dy
        delta = (partialAx(x,y),partialAy(x,y))
        print("coordinates: (" + str(x) + "," + str(y) + ") vector: " + str(delta))
    print((x,y))
elif sys.argv[1] == "B":
    delta = (partialBx(0,0),partialBy(0,0))
    x,y = start 
    while magnitude(delta) > math.pow(10,-8) or -1 * magnitude(delta) > math.pow(10,-8):
        dx,dy = delta 
        grad = make_grad(B,x,y,dx,dy) #Line optimize 
        rate = one_d_minimize(grad,0,1,math.pow(10,-8)) #Line optmize 
        print("learning rate: " + str(rate))
        x = x - rate * dx
        y = y - rate * dy
        delta = (partialBx(x,y),partialBy(x,y))
        print("coordinates: (" + str(x) + "," + str(y) + ") vector: " + str(delta))
    print((x,y))

