import numpy as np

#questo .py viene usato per splittare il dataset


def split_data(x,y, tr_fraction=0.5):

    n,d = x.shape   #prendiamo il numero di immagini e la dimensone di un immagine
    n1 = y.size     #numero di classi (di cifre)

    assert (n == n1)    #solleva un eccezzione AssertionError

    #lavoriamo sugli indici per risparmiare
    n_tr = int(np.round(n * tr_fraction)) #dividiamo n per splittare
    idx = np.array(range(0, n))           #creo un array che va da 0 a n-1
    np.random.shuffle(idx)                #randomizziamo i campioni
    idx_tr = idx[0:n_tr]                  #da 0 a n_tr-1  TRAIN
    idx_ts = idx[n_tr:n]                  #da n_tr a n-1  TEST

    #lavoriamo ora sui dati
    xtr = x[idx_tr, :]
    ytr = x[idx_tr]
    xts = x[idx_ts, :]
    yts = y[idx_ts]

    return xtr, ytr, xts, yts