


"""
    def test_predict(self): #se non è fittato, crea un exeption
        self.assertRaises(ValueError, self.clf.predict,self.x)
        self.clf.fit(self.x,self.y)
        ypred = self.clf.predict(self.x)
        self.assertequal(ypred.shape,self.y.shape)
        self.assertequal(np.sum(ypred==self.y),self.n_samples) #con == ottengo un vettore che dovrebbe essere tutto 1, se sommo tutto dovrei avere il numero delle righe, ovvero il numero dei samples

        prob = self.clf.decision_funciton(self.x, softmax_scaling = True)
        np.sum(probs, axis=1)
"""