
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib

file_path = pathlib.Path(__file__).parent.absolute()

class Network(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:

        super().__init__()
        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cuda'
        # TODO: Calcular dimension de salida
        out_dim = self.calc_out_dim(input_dim,kernel_size= 5)
        #print("Out Dim:", out_dim)#para dos capas 40, tres 36...
        torch.backends.cuda.matmul.allow_tf32 = True #Se supone que funciona solo para RTX 30XX, activa cores especificos para hacer convoluciones mas rapido
        # TODO: Define las capas de tu red

        self.layers =nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(128, 178, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(178, 200, kernel_size=5),
            nn.Flatten(start_dim = 1),
            nn.Linear(200 * 36 * 36, 1024),
            nn.ReLU(),
            nn.Linear(1024,n_classes)#Esta red tarda ~1 hora por 5 epoch en entrenar [1024, 7]
        )#Best_model1
        
        """
        self.conv1 = nn.Conv2d(1,16 ,kernel_size= 5)
        self.conv2 = nn.Conv2d(16,32 ,kernel_size= 5)
        self.lineal1 = nn.Linear(32*40*40,1024)
        self.lineal2 = nn.Linear(1024,n_classes)
        """#ejemplo del CNN del jupyter
        
        self.to(self.device)
 
    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = math.floor((in_dim - kernel_size + 2*padding)/stride) + 1
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Define la propagacion hacia adelante de tu red
        #print("Tensor X ",x.shape)
        logits =self.layers(x)
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        #print("Before Flatten Shape:", x.shape)
        x = torch.flatten(x,start_dim= 1)
        #print("After Flatten Shape:", x.shape)
        x = self.lineal1(x)
        x = F.relu(x)
        x = self.lineal2(x)
        logits = x
        """
        proba = F.softmax(logits, dim = 1)#se supone que linear ya aplica softmax al final

        return logits , proba

    def predict(self, x):
        with torch.inference_mode():
            return self.forward(x)

    def save_model(self, model_name: str):
        '''
            Guarda el modelo en el path especificado
            args:
            - net: definición de la red neuronal (con nn.Sequential o la clase anteriormente definida)
            - path (str): path relativo donde se guardará el modelo
        '''
        models_path = file_path / 'models' / model_name
        # TODO: Guarda los pesos de tu red neuronal en el path especificado
        torch.save( self.state_dict(), models_path )

    def load_model(self, model_name: str):
        '''
            Carga el modelo en el path especificado
            args:
            - path (str): path relativo donde se guardó el modelo
        '''
        # TODO: Carga los pesos de tu red neuronal
        models_path = file_path / 'models' / model_name
        self.load_state_dict(torch.load(models_path))

