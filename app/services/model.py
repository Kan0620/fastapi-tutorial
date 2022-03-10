'''
Copyright (c) 2019 Timothy Esler
Released under the MIT license
https://github.com/timesler/facenet-pytorch/blob/master/LICENSE.md
'''
from abc import ABC, abstractmethod
from typing import Any

from io import BytesIO
import base64
import glob

from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms
import numpy as np

from .my_resenet import MyInceptionResnetV1

class BaseMLModel(ABC):
    @abstractmethod
    def cut(self, req: Any) -> Any:
        raise NotImplementedError
    def predict(self, req: Any) -> Any:
        raise NotImplementedError


class MLModel(BaseMLModel):
    """Sample ML model"""

    def __init__(self, model_path: str) -> None:
        self.mtcnn = MTCNN(image_size = 160, margin = 10).eval()
        #self.resnet = MyInceptionResnetV1(pretrained = 'vggface2').eval()
        #self.resnet.load_state_dict(torch.load("./services/resnet"))
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.ids = np.load("data/id.npy")
        self.vecs = np.load("data/actress_vecs.npy", allow_pickle = True)

        
    def cut(self, input_text: str) -> dict:
        img = Image.open(BytesIO(base64.b64decode(input_text)))
        with torch.no_grad():
            img = self.mtcnn(img, "img.png")
        is_face = True
        if str(img) == "None":
            is_face = False

        return {"is_face": is_face, "img": str(img)}

    def predict(self, input_text: str) -> dict:
        img = Image.open(BytesIO(base64.b64decode(input_text)))
        img = self.transform(img).reshape((1, 3, 160, 160))
        with torch.no_grad():
            img = self.resnet(img)
        img = img[0].detach().numpy()
        rec_actress_id = []
        for index in np.argsort(np.square(self.vecs - img).sum(axis = 1))[:10]:
                rec_actress_id.append(str(self.ids[index]))

        return {"rec_actress_id": rec_actress_id}
