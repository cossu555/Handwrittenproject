import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from splitters import split_data
from classifiers import NMC

data = pd.read_csv("data/mnist_data.csv")
data = np.array(data)
print(data.shape)
print(type(data))

