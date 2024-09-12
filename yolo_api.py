import cv2 
from ultralytics import YOLO


def detect(frame, detection_threshold, rect_color, model_name = 'YOLOv9 Tiny'):
    # Get class names 
    model = load_model (model_name)
    class_names = model.names
    people_found = False
    result = model (frame, save=False, verbose=False) ## Inference 

    if len (result[0]): # If no objects have been detected 
        for i in range (len (result[0].boxes.xyxy)): # Iterate over the objects in a single image
            cls_idx = result[0].boxes.cls[i]
            cls_name = class_names[int (cls_idx)]
            if cls_name != 'person' or result[0].boxes.conf[i] < detection_threshold : continue
            people_found = True

            x1, y1, x2, y2 = map(int, result[0].boxes.xyxy[i].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 2)
            
    return people_found


    
models = {'YOLOv9 Tiny': 'models\\yolov9t.pt', 'YOLOv10 Nano' : 'models\\yolov10n.pt', 'YOLOv10 Small' : 'models\\yolov10s.pt', 'YOLOv10 Medium' : 'models\\yolov10m.pt', 'YOLOv10 Big' : 'models\\yolov10b.pt', 'YOLOv10 Large' : 'models\\yolov10l.pt', 'YOLOv10 Extra Large' : 'models\\yolov10x.pt'}
def load_model (model_name='YOLOv9 Tiny'):
    return YOLO (models[model_name])