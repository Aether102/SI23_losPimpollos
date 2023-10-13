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
1. Reducimos la dimensionalidad de los datos de validacion **data_val** a 2 dimensiones usando TSNE y PCA
```
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Usando PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_val)

# Usando t-SNE
tsne = TSNE(n_components=2)
data_tsne = tsne.fit_transform(data_val)
``` 
2. Graficar nuevos datos uno al lado del otro
   
![pca_and_tsne](https://github.com/DiegoMarin11/SI23_losPimpollos/assets/108961521/7597b5b8-262b-4740-aad8-edb0f00b561d)

**¿Cual método de reducción de dimensionalidad funciona mejor en este caso?**
**¿Que puedes deducir de esta imagen?**

En este caso, t-SNE nos permite observar mas claramente los clusters dentro de los datos.

**¿Qué representa cada color en este caso?**

Cada color en los diagramas de dispersión representa un dígito diferente (0-9) en el conjunto de datos.

## Seccion 2: Entrenamiento

### Preprocesamiento

La estandarización se realiza utilizando StandardScaler de scikit-learn, que estandariza el conjunto de datos para que tenga media=0 y varianza=1.
```
scaler = StandardScaler()
scaler.fit(data_train)
```

### Modelos de entrenamiento

Utilizaremos tres modelos diferentes:

**1. K-Nearest Neighbors (KNN) (aprendizaje supervisado)**

- Aprendizaje no paramétrico: KNN no hace suposiciones sobre la distribución de datos y no utiliza parámetros para predecir. Utiliza directamente los datos de entrenamiento durante la predicción, lo que puede resultar útil cuando el límite de decisión es irregular.
- Clasificación multiclase: KNN naturalmente admite la clasificación multiclase, lo que la hace apta para la clasificación de dígitos donde hay múltiples clases (0-9).

**2. Support Vector Machines (SVM) (aprendizaje supervisado)**

- *Linear: especifica un kernel lineal*
- *RBF:  especifica un kernel de función de base radial (RBF), que es una opción popular para problemas no lineales. El parámetro gamma define hasta dónde llega la influencia de un único ejemplo de entrenamiento y se puede establecer en "escala", "automático" o un valor flotante.*
- Efectivo en espacios de alta dimensión: dado que los datos de imagen pueden ser de alta dimensión, SVM es capaz de manejar esto a través de su efectividad en espacios de alta dimensión.
- Truco del kernel: la capacidad de utilizar diferentes funciones del kernel permite a SVM resolver problemas no lineales, lo que puede ser beneficioso si los dígitos forman grupos separables no linealmente en el espacio de características.

**3. K-Means Clustering (aprendizaje no supervisado)**

- Aprendizaje no supervisado: K-Means es fundamentalmente un algoritmo de aprendizaje no supervisado, que normalmente se utiliza para agrupar en lugar de clasificar. Fue elegido aquí para ilustrar los desafíos y resultados del uso de un algoritmo de agrupamiento para una tarea de clasificación.
- Exploración de datos: a veces puede revelar patrones o agrupaciones interesantes en los datos que podrían no ser evidentes con los datos etiquetados.
 
El uso de diferentes tipos de algoritmos permite realizar un análisis comparativo sólido. Estos algoritmos sirven como buenos puntos de referencia debido a su popularidad y amplio uso en la comunidad de aprendizaje de maquina.

### Procedimiento
1. Importar clases requeridas
```
from sklearn.neighbors import KNeighborsClassifier # K Nearest Neigbors
from sklearn.svm import SVC                        # Support Vector Classifier
from sklearn.cluster import KMeans                 # K Means
```
2. Entrena el modelo y regresa modelo entrenado
```
def train(X, label, model_type:str):
...
    if model_type == "knn":
        estimator = KNeighborsClassifier(n_neighbors=3)
    elif model_type == "svm_linear":
        estimator = SVC(kernel='linear')
    elif model_type == "svm_rbf":
        estimator = SVC(kernel='rbf', gamma='scale')
    elif model_type == "kmeans":
        estimator = KMeans(n_clusters=len(np.unique(label)))  # Numero de clusters igual al numero de digitos
...
return estimator
```

## 3. Evaluación y análisis de las predicciones

Graficar los datos de VALIDACIÓN reducidos
```
    for i, reduced_data, title in zip(range(2), [data_pca, data_tsne], ["PCA", "t-SNE"]):
        for g in groups:
            # TODO: Grafica los datos de VALIDACIÓN reducidos (reduced_data.shape = (N, 2))
            # Tal que grafiques aquellos que correspondan al grupo/clase group
            # Investiga plt.scatter, np.where o cómo filtrar arreglos dada una condición booleana
            mask = (preds ==g)
            ax[i].scatter(reduced_data[mask, 0], reduced_data[mask, 1], label=f"Group {g}")
        
        ax[i].legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
```

### **K-Nearest Neighbor**
![knn_test](https://github.com/DiegoMarin11/SI23_losPimpollos/assets/108961521/4aa90342-c65e-4ae3-bd83-09ea56b01598)
### **SVM Linear**


### **SVM Radial**


### **K-Means**

