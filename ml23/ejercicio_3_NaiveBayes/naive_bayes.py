import numpy as np

class NaiveBayes():
    def __init__(self, alpha=1) -> None:
        self.alpha = 1e-10 if alpha < 1e-10 else alpha

    def fit(self, X, y):
        # TODO: Calcula la probabilidad de que una muestra sea positiva P(y=1)
        self.prior_positives = sum(y)/len(y)

        # TODO: Calcula la probabilidad de que una muestra sea negativa P(y=0)
        self.prior_negative = 1 - self.prior_positives

        # TODO: Para cada palabra del vocabulario x_i
        # calcula la probabilidad de: P(x_i| y=1)
        # Guardalas en un arreglo de numpy:
        # self._likelihoods_positives = [P(x_1| y=1), P(x_2| y=1), ..., P(x_n| y=1)]
        self._likelihoods_positives = np.zeros(X.shape[1])
        self._likelihoods_negatives = np.zeros(X.shape[1])

        for i in range(len(y)):
            if y[i] == 1:
                self._likelihoods_positives[i]+=X[i]
            else:
                self._likelihoods_negatives[i]+=X[i]


        self._likelihoods_positives = (self._likelihoods_positives + self.alpha) / (sum(y == 1)+(X.shape[1] * self.alpha)) #sum(y==1)
        
        # TODO:  Para cada palabra del vocabulario x_i, calcula P(x_i| y=0)
        # Guardalas en un arreglo de numpy:
        # self._likelihoods_negatives = [P(x_1| y=0), P(x_2| y=0), ..., P(x_n| y=0)]

        self._likelihoods_negatives = (self._likelihoods_negatives + self.alpha) / (sum(y == 0)+(X.shape[1] * self.alpha))
        return self

    def predict(self, X):
        # TODO: Calcula la distribución posterior para la clase 1 dado los nuevos puntos X
        # utilizando el prior y los likelihoods calculados anteriormente
        # P(y = 1 | X) = P(y=1) * P(x1|y=1) * P(x2|y=1) * ... * P(xn|y=1)
        posterior_positive =  self.prior_positives * np.prod(self._likelihoods_positives[X == 1]) * np.prod(1 - self._likelihoods_positives[X == 0])

        # TODO: Calcula la distribución posterior para la clase 0 dado los nuevos puntos X
        # utilizando el prior y los likelihoods calculados anteriormente
        # P(y = 0 | X) = P(y=0) * P(x1|y=0) * P(x2|y=0) * ... * P(xn|y=0)
        posterior_negative = self.prior_negative * np.prod(self._likelihoods_negatives[X == 1]) * np.prod(1 - self._likelihoods_negatives[X == 0])

        # TODO: Determina a que clase pertenece la muestra X dado las distribuciones posteriores
        if posterior_positive > posterior_negative:
            clase = 1
        else:
            clase = 0
            
        return clase
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / len(y)