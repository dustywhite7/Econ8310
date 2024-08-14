import numpy as np
import pandas as pd
import patsy as pt

data = pd.read_csv("/home/dusty/DatasetsDA/Titanic/train.csv")

y, x = pt.dmatrices("Survived ~ -1 + C(Sex) + Age + SibSp", data=data)


# Given the probability of each outcome, calculate the nat-entropy of the system

def natEnt(listP):
    entropy = 0
    n = len(listP)
    for i in range(n):
        entropy += listP[i] * np.log(listP[i])
    entropy *= -1
    return np.abs(entropy)
    

# Determine the number of classes in dep. var, and calculate the probability of each
    
def prob(depVar):
    vals = np.unique(depVar)
    k = len(vals)
    n = len(depVar)
    counts = [0]*k
    for i in range(n):
        for j in range(k):
            if (depVar[i]==vals[j]):
                counts[j]+=1
    for i in range(k):
        counts[i] /= n
    return counts


def searchFn(y, x):
    minx = np.min(x)
    maxx = np.max(x)
    base = natEnt(prob(y))
    ent = [0]*len(y)
    bestx = minx
    gain = 0
    for i in np.linspace(minx, maxx, num=len(np.unique(y))):
        a = y[(x>=i)]
        b = y[(x<i)]
        if ((natEnt(prob(a)*len(a)) + natEnt(prob(b))*len(b)) / len(y) < base):
            bestx = i
            gain = base - (natEnt(prob(a)*len(a)) + natEnt(prob(b))*len(b)) / len(y)
    return gain, bestx
        


# Determine the x variable that will most reduce entropy

def bestGain(y, x):
    base = natEnt(prob(y))
    best = [0,0,0,None]
    for i in range(np.shape(x)[1]):
        if (searchFn(y,x[:,i])[0] > best[0]):
            best = [searchFn(y,x[:,i])[0], searchFn(y,x[:,i])[1], i, None]
    return best


def cart(y, x, depth=3, least=5):
    tree = list()
    count = depth
    while count>0:
        if (len(tree)<1):
            tree.append(bestGain(y,x))
        else:
            
    

