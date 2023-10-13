# Proyecto 1: Identificando Digitos

## Descripción General
Este script realiza la clasificación en el conjunto de datos de dígitos (MNIST), utilizando varios modelos de aprendizaje automático como K-Nearest Neighbors (KNN), Support Vector Machine (SVM), y K-Means Clustering. El conjunto de datos se preprocesa, los modelos se entrenan y las predicciones se visualizan y evalúan.

### Dependencias
- numpy
- matplotlib
- sklearn

## Seccion 1: Analizando los datos
### Visualización en baja dimensionalidad
La utilización de PCA y t-SNE para la reducción de dimensionalidad nos permite comparar e interpretar visualmente el conjunto de datos de alta dimensión en un espacio 2D uno al lado del otro.
1. Aplicar PCA y t-SNE
```
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# TODO Reducimos la dimensionalidad de los datos de validacion data_val
# a 2 dimensiones usando TSNE y PCA

# Usando PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_val)

# Usando t-SNE
tsne = TSNE(n_components=2)
data_tsne = tsne.fit_transform(data_val)
``` 
