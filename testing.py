import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.optimize import minimize
import cvxpy as cp

#import dictionary
from marc.LSTM import dictWeightedAssets
with open("marc\selected_assets test1.json") as file:
    data=json.load(file)

dictAssets=dictWeightedAssets(data)

#parameters to tune
samples=1 #sample random weights
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

#test with all equal weights
equalguad=[]
meanweight=1/n
equalguad=np.sum(np.dot(differenza, meanweight))

ritorni = differenza 

# Funzione obiettivo da minimizzare (negativo del guadagno totale)
def obiettivo(pesi):
    guadagno_totale = np.sum(pesi * ritorni)
    return -guadagno_totale  # Minimizzare il negativo del guadagno per massimizzarlo

# Restrizioni: la somma dei pesi deve essere 1 e i pesi devono essere >= 0
vincolo_somma = {'type': 'eq', 'fun': lambda pesi: np.sum(pesi) - 1}
vincoli_min = [{'type': 'ineq', 'fun': lambda pesi, i=i: pesi[i] - 0.01} for i in range(n)]
vincoli_max = [{'type': 'ineq', 'fun': lambda pesi, i=i: 0.40 - pesi[i]} for i in range(n)]

restrizioni = [vincolo_somma] + vincoli_min + vincoli_max

# Condizioni iniziali (distribuzione uguale tra gli asset)
pesi_iniziali = np.full(n, 1/n)

# Risoluzione del problema di ottimizzazione
risultato = minimize(obiettivo, pesi_iniziali, constraints=restrizioni)

# Pesi ottimali
pesi_ottimali = risultato.x

# Calcolare il guadagno totale con i pesi ottimali
guadagno_totale = np.sum(pesi_ottimali * ritorni)

# Risultati
print("Pesi ottimali per massimizzare il guadagno:", pesi_ottimali)
print("Guadagno totale:", guadagno_totale)




print("untrained:",valcas)
print("equal:", equalguad)
print("trained:",guadagno)