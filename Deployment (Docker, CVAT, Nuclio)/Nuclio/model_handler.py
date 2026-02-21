import os
import cv2
import numpy as np
import torch
from torchvision import transforms
import atomai as aoi

def to_cvat_mask(box: list, mask):
    xtl, ytl, xbr, ybr = box
    flattened = mask[ytl:ybr + 1, xtl:xbr + 1].flat[:].tolist()
    flattened.extend([xtl, ytl, xbr, ybr])
    return flattened

class ModelHandler:
    def __init__(self, labels):
        self.labels = labels
        self.model = self._load_model()

    def _load_model(self):
        # Initialize your Segmentor model
        new_model = aoi.models.Segmentor(nb_classes=1)
        #new_model = aoi.models.Segmentor(model='dilnet',nb_classes=1) 
        file_path = 'model_new.pth'

        if os.path.isfile(file_path):
            print(f"{file_path} exists and is a file.")
        else:
            print(f"{file_path} either does not exist or is not a file.")
        
        # Load weights into the model
        new_model.load_weights(file_path)

        return new_model
    
    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        image = transform(image)

        return image
    
    def infer(self, image, threshold=0.5):

        img_width = image.width
        img_height = image.height

        image = np.array(image) #Converting PIL image to numpy array
        
        pred = self.model.predict(np.expand_dims(image, axis=0))[0][0]
        binary_mask = (pred >= threshold).astype(np.uint8)

        width, height, _ = binary_mask.shape

        results = []

        for i in range(len(self.labels)):
            mask_by_label = np.zeros((width, height), dtype=np.uint8)
            mask_by_label = ((binary_mask == (float(i)+1.0)) * 255).astype(np.uint8)
            mask_by_label = cv2.resize(mask_by_label,
                dsize=(img_width, img_height),
                interpolation=cv2.INTER_NEAREST)

            contours, _ = cv2.findContours(mask_by_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                contour = np.flip(contour, axis=1)
                if len(contour) < 3:
                    continue

                x_min = max(0, int(np.min(contour[:,:,0])))
                x_max = max(0, int(np.max(contour[:,:,0])))
                y_min = max(0, int(np.min(contour[:,:,1])))
                y_max = max(0, int(np.max(contour[:,:,1])))

                cvat_mask = to_cvat_mask((x_min, y_min, x_max, y_max), mask_by_label)

                results.append({
                    "confidence": None,
                    "label": self.labels.get(i, "unknown"),
                    "points": contour.ravel().tolist(),
                    "mask": cvat_mask,
                    "type": "mask",
                })

        return results