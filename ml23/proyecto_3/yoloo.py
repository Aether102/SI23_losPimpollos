from ultralytics import YOLO
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import cv2
import torch
import pandas as pd
from PIL import Image
device = torch.device('cuda') #Corremos la inferencia en CUDA para mayor rapidez

def get_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model.to(device)
    model.conf = 0.60  # NMS confidence threshold
    model.iou = 0.60  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    model.max_det = 5  # maximum number of detections per image
    model.amp = False  # Automatic Mixed Precision (AMP) inference
    return model



camara = cv2.VideoCapture(0)


model = get_model()

while True:
    ret, frame = camara.read()
    #preprocesamiento de imagen
    
    input_tensor = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = model(input_tensor)
    #print("resultado = ",result.shape)
    for det in result.pandas().xyxy[0].iterrows():# xmin  /  ymin  /  xmax /  ymax / confidence / class  /  name
        _, det = det  # Obtener la fila como una Serie de pandas
        x_min, y_min, x_max, y_max, conf, clase, nombre = det.values 
        label = nombre# = name
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        cv2.putText(frame, f'Clase: {label}', (int(x_min), int(y_min) -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)#-10 para que la etiqueta no se solape con el cuadro
        cv2.imshow('Deteccion en tiempo real', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
camara.release()
cv2.destroyAllWindows()
