import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from splitters import split_data
from classifiers import NMC

#ricorda che COMMIT è LOCALE, PUSH lo invia al server

data = pd.read_csv("../data/mnist_data.csv") #i .. servono per andare nella cartella padre
data = np.array(data)
print(data.shape)
print(type(data))

y= data[:, 0]   #di tutti i campioni prende i y (0)
x= data[:, 1:]  #di tutti i campioni prendo l'immagine (1,785)

idx = 100
plt.imshow(x[idx, :].reshape(28,28), cmap="gray") #visualizziamo l'immagine numero 100, facendo un reshape 28*28
plt.show()
print(y[idx]) #printiamo l'etichetta (la classe)

xtr, ytr, xts, yts = split_data(x, y, tr_fraction=0.6)
print(xtr.shape, ytr.shape)
#splittiamo i dati: TRAIN e TEST

#prnediamo 10 zero
xk = xtr[ytr == 0, :]               #prendo solo le immagini dello 0 zero
plt.figure()                        #creo la figura
for i in range(10):                 #ne prendo solo 10
    plt.subplot(2, 5, i + 1)
    plt.imshow(xk[i, :].reshape(28, 28), cmap='gray')
plt.show()
print(xk.shape)


#prendiamo la media di pixel dei 10 zero
meank = np.mean(xk, axis=0)
plt.figure()
plt.imshow(meank.reshape(28, 28), cmap='gray')
plt.show()

# creiamo una istanza NMC classifier (una ML)
clf = NMC()
clf.fit(xtr, ytr) #la addestriamo

centroids = clf.centroids #prendiamo i centroids

#clf.centroids = 1 #modifichiamo i centroids nella variabile 1

#vediamo tutti  centroidis addesrati
plt.figure()
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(centroids[i, :].reshape(28, 28), cmap='gray')
plt.show()

print(centroids.shape)
print(xts.shape)

ypred = clf.predict(xts)

accuracy = np.mean(ypred == yts)
print("Accuracy:", accuracy)