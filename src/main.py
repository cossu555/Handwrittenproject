import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from splitters import split_data
from classifiers import NMC

data = pd.read_csv("../data/mnist_data.csv") #i .. servono per andare nella cartella padre
data = np.array(data)
print(data.shape)
print(type(data))

y= data[:, 0]   #di tutti i campioni prende i y (0)
x= data[:, 1:]  #di tutti i campioni prendo l'immagine (1,785)

