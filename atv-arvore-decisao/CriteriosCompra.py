import pandas as pd
import numpy as np
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from scipy.io import arff

data, meta = arff.loadarff('./ComprasProduto.arff')

attributes = meta.names()
data_value = np.asarray(data)

idade = np.asarray(data['Idade']).reshape(-1, 1)
renda_mensal = np.asarray(data['RendaMensal']).reshape(-1, 1)
tempo_navegacao = np.asarray(data['TempoNavegacao']).reshape(-1, 1)

features = np.concatenate((idade, renda_mensal, tempo_navegacao), axis=1)

target = data['Compra']

Arvore = DecisionTreeClassifier(criterion='entropy').fit(features, target)

plt.figure(figsize=(10, 6.5))
tree.plot_tree(Arvore, feature_names=['Idade', 'RendaMensal', 'TempoNavegacao'], class_names=['Sim', 'Nao'],
                   filled=True, rounded=True)
plt.show()

fig, ax = plt.subplots(figsize=(25, 10))
metrics.ConfusionMatrixDisplay.from_estimator(Arvore, features, target, display_labels=['Sim', 'Nao'], values_format='d', ax=ax)
plt.show()
