import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

#import dictionary
from marc.LSTM import dictWeightedAssets
with open("marc\selected_assets test1.json") as file:
    data=json.load(file)

dictAssets=dictWeightedAssets(data)

#parameters to tune
samples=1000 #sample random weights
n= 20 #numero asset

#variables definition
ultimovalore=[]
penultimovalore=[]
nomi = list(dictAssets.keys())
pesi = list(dictAssets.values())
pesi = [float(v) for v in pesi]

#find final values of assets
for i in range(n):
    storico=data[nomi[i]]['history']
    ultimadata = sorted(storico.keys())[-1]
    ultimovalore.append(storico[ultimadata])

#find second to last values of assets
for j in range(n):
    storico=data[nomi[j]]['history']
    penultimadata = sorted(storico.keys())[-2]
    penultimovalore.append(storico[penultimadata])

#find difference and gain for trained weights
ultimovalore = np.array(ultimovalore)
penultimovalore = np.array(penultimovalore)
differenza=ultimovalore-penultimovalore
guadagno=[]
guadagno=np.sum(np.dot(differenza,pesi))

#find gain for untrained weights
pocoguad=[]

for l in range(samples):
    randweights = np.random.dirichlet(np.ones(n))
    pocoguad.append(np.sum(np.dot(randweights,differenza)))
    
valcas=np.mean(pocoguad)

print("untrained:",valcas)
print("trained:",guadagno)