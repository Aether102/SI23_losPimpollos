import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from network import Network
import torch
from utils import to_numpy, get_transforms, add_img_text
from dataset import EMOTIONS_MAP
import pathlib

file_path = pathlib.Path(__file__).parent.absolute()

def load_img(path):
    assert os.path.isfile(path), f"El archivo {path} no existe"
    img = cv2.imread(path)
    val_transforms, unnormalize = get_transforms("test", img_size = 48)
    tensor_img = val_transforms(img)
    denormalized = unnormalize(tensor_img)
    return img, tensor_img, denormalized

def predict(img_title_paths):
    '''
        Hace la inferencia de las imagenes
        args:
        - img_title_paths (dict): diccionario con el titulo de la imagen (key) y el path (value)
    '''
    # Cargar el modelo
    modelo = Network(48, 7)
    modelo.load_model("best_model1.pt")
    for path in img_title_paths:
        # Cargar la imagen
        # np.ndarray, torch.Tensor
        im_file = (file_path / path).as_posix()
        original, transformed, denormalized = load_img(im_file)

        # Inferencia
        #print(transformed.cuda().shape)#[256,1 ,48, 48]
        transformed = transformed.unsqueeze(0).cuda()# Agrega una dimension porque en training el modelo recibe [256, 1, 48, 48]
        logits, proba = modelo.predict(transformed)
        pred = torch.argmax(proba, -1).item()
        pred_label = EMOTIONS_MAP[pred]

        # Original / transformada
        h, w = original.shape[:2]
        resize_value = 300
        img = cv2.resize(original, (w * resize_value // h, resize_value))
        img = add_img_text(img, f"Pred: {pred_label}")

        # Mostrar la imagen
        denormalized = to_numpy(denormalized)
        denormalized = cv2.resize(denormalized, (resize_value, resize_value))
        cv2.imshow("Predicción - original", img)
        cv2.imshow("Predicción - transformed", denormalized)
        cv2.waitKey(0)

if __name__=="__main__":
    # Direcciones relativas a este archivo
    #img_paths = ["./test_imgs/ivan.png"]
    #Abrimos todas las imagenes que se encuentren en la carpeta test_imgs para predecir cada una
    test_imgs_folder = os.path.join(file_path, "test_imgs")
    img_paths = [os.path.join(test_imgs_folder, f) for f in os.listdir(test_imgs_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    predict(img_paths)
    
