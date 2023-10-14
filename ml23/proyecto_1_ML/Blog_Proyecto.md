# Proyecto 1: Identificando Digitos

## Descripción General
Este script realiza la clasificación en el conjunto de datos de dígitos, utilizando varios modelos de aprendizaje automático como K-Nearest Neighbors (KNN), Support Vector Machine (SVM), y K-Means Clustering. El conjunto de datos se preprocesa, los modelos se entrenan y las predicciones se visualizan y evalúan.

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
### 3.1 (Inferencia) Datos de validación en baja dimensionalidad

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
![svm_linear](https://github.com/DiegoMarin11/SI23_losPimpollos/assets/108961521/bbcc939a-ffad-4cdc-b3e9-f40ecdf69f92)
### **SVM Radial**
![svm_rbf](https://github.com/DiegoMarin11/SI23_losPimpollos/assets/108961521/48c1952f-b02e-4f00-bfc1-1beb88ee5b42)
### **K-Means**
![kmeans](https://github.com/DiegoMarin11/SI23_losPimpollos/assets/108961521/78ca7f66-7514-4fb6-8bca-cecb13f44f49)

### 3.2 (Inferencia) Visualizar imagenes en cada grupo/clase

El siguiente código llama al método de inferencia anteriormente definido.
```
def vis_preds(trained_model, data_val, target_val, model_name):
    preds = inference(modelo, data_val)
    group_pred = np.unique(preds)
    n_groups = len(group_pred)
...
    for group, ax in zip(group_pred, axes):
        #======================= Start  solution=====================
        # TODO: Filtra data_val para quedarte solamente con aquellos elementos
        # donde la predicción de tu modelo sea igual a group
        mask = (preds == group)
        filtered_data_val = data_val[mask]
        filtered_target_val = target_val[mask]

        # TODO: Selecciona una imagen de los datos en data_val donde pred == group
        # y selecciona la etiqueta real para dicha imagén para mostrarlos juntos
        random_idx = np.random.choice(len(filtered_data_val))
        img_vector = filtered_data_val[random_idx]
        gt = filtered_target_val[random_idx]

        # TODO: Calcula la predicción del modelo para la imagen aleatoria
        # usando el modelo entrenado "trained_model"
        pred = inference(trained_model, img_vector)

        # TODO: Cambia la forma de la imagen usando np.reshape a (8, 8)
        img = img_vector.reshape((8, 8))
        
        # TODO: Visualiza la imagen de 8x8 usando ax.matshow Similar al inicio del ejercicio
        ax.matshow(img, cmap=plt.cm.gray)
...
plt.show()
```

### Resultados
### **K-Nearest Neighbor**
![knn_visual](https://github.com/DiegoMarin11/SI23_losPimpollos/assets/108961521/830697cb-366a-4137-b2aa-3e047111b704)

### **SVM Linear**
![svm_lin_visual](https://github.com/DiegoMarin11/SI23_losPimpollos/assets/108961521/28f60b40-3ad1-46c8-80ee-0d62b7afbe18)

### **SVM Radial**
![svm_rbf_visual](https://github.com/DiegoMarin11/SI23_losPimpollos/assets/108961521/21309955-7929-42bc-9d07-7172a92d3af4)

### **K-Means**
![kmenas_visual](https://github.com/DiegoMarin11/SI23_losPimpollos/assets/108961521/b9641a02-c929-4120-ab26-e1ad8c7d0629)

### 3.2 (Inferencia) Coomparar rendimiento de distintos modelos

En esta sección se evalúa los modelos entrenados en el conjunto de validación utilizando alguna métrica vista en clase (accuracy, F1, Precision, Recall etc.) y se determina cuantitativamente cual funciona mejor. 

En este proyecto, se emplearon varias métricas de clasificación para evaluar cuantitativamente y comparar el rendimiento de diferentes modelos de aprendizaje(KNN, SVM y K-Means) en la tarea de clasificación de dígitos. Esto facilitó una comparación directa de los resultados, lo que permitió identificar el modelo que funcionó de manera óptima y proporcionó información valiosa sobre las fortalezas y debilidades de cada enfoque.

1. Importar clases
```
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
```
2. Calcula la predicción y evalúa la calidad de predicción vs. las etiquetas reales (target_val)
```
def evaluate_model(trained_model, data_val, target_val, model_name):
    # predicciones
    preds = inference(trained_model, data_val)
    
    accuracy = accuracy_score(target_val, preds)
    precision = precision_score(target_val, preds, average='weighted')
    recall = recall_score(target_val, preds, average='weighted')
    f1 = f1_score(target_val, preds, average='weighted')
```
**Resultados**

<img width="144" alt="metricsknn" src="https://github.com/DiegoMarin11/SI23_losPimpollos/assets/108961521/f336bde2-5bad-49af-81c9-ac30a74b5219">
<img width="148" alt="metricssvmlinear" src="https://github.com/DiegoMarin11/SI23_losPimpollos/assets/108961521/eda01249-0c89-4163-9c25-9cf083ef9b9e">


<img width="141" alt="svmrbf metrica" src="https://github.com/DiegoMarin11/SI23_losPimpollos/assets/108961521/924c4b8a-20e2-4662-8423-227e37c22038">
<img width="142" alt="kmenas metrics" src="https://github.com/DiegoMarin11/SI23_losPimpollos/assets/108961521/5b3be906-58c3-4ea8-9555-5c61d2591a19">

## Conclusion

En resumen, mantener nuestros datos en alta dimensionalidad fue clave para conservar detalles importantes que ayudan a distinguir los dígitos. Elegimos el aprendizaje supervisado, utilizando métodos como KNN y SVM, porque utiliza ejemplos conocidos para hacer predicciones futuras, lo cual es útil para clasificar cosas en categorías, como identificar dígitos. También era vital asegurarnos de que todos los datos de nuestras imágenes estuvieran normalizados o ajustados a una escala común. Este paso aseguró que ninguna parte de nuestros datos tuviera demasiada influencia sobre nuestros modelos solo por su tamaño, lo que ayudó a que nuestros modelos aprendieran de manera justa de todas las partes de los datos y mantuvo nuestro entrenamiento estable. 
