import sys
from random import *
import numpy as np
from math import *
import time
from matplotlib import pyplot as plt
class  Neural_Network(object):
    def __init__(self):
        self.inputlayers=2
        self.outputlayers=1
        self.hiddenlayers=3
        self.w1=np.random.randn(self.inputlayers,self.hiddenlayers)
        self.w2=np.random.randn(self.hiddenlayers,self.outputlayers)
    def forward(self,X):
        self.z2=np.dot(X,self.w1)
        self.a2=self.myactfun(self.z2,2)
        self.z3=np.dot(self.a2,self.w2)
        yvect=self.myactfun(self.z3,3)
        return yvect
    def sigmoid(self,Z):
        return 1/(1+np.exp(-Z))
    def myactfun(self,Z,n):
        sm=0.0
        for i in range(0,n+1):
            sm+=pow(pi,i+1)*np.exp(-i*Z)
        return 1/sm
X=np.array(([3,5],[5,1],[6,2]),dtype=int)
nn=Neural_Network()
yvect=nn.forward(X)
print("\nThe final precised vector is:\t"+str(yvect))
weights=np.linspace(-10,10,1000)
costs=np.zeros(1000)
starttime=time.clock()
y=np.array([[0.75],[0.82],[0.93]])
print("Our actul assumption is:\t"+str(y))
accur=0.0
for i in range(1000):
    nn.w1[0,0]=weights[i]
    yvect=nn.forward(X)
    costs[i]=0.5*sum((y-yvect)**2)
    accur+=sum(y-yvect)/sum(y)
print(costs)
endtime=time.clock()
timeelapsed=endtime-starttime
print(timeelapsed)
x=np.array([[0.5],[1],[1.5]])
print("\n The minimum cost is:\t"+str(min(costs)))
print("\n The accuracy is "+str(accur/10))
