import io
import torch
from PIL import Image,ImageOps
from torchvision import models,transforms
from config import CLASSES,WEIGHTS
T=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
def create_model():
    m=models.mobilenet_v3_small(weights=None)
    m.classifier[-1]=torch.nn.Linear(m.classifier[-1].in_features,len(CLASSES))
    return m
def load_model():
    ckpt=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
    if ckpt['classes']!=CLASSES: raise ValueError('Class ordering differs from checkpoint')
    m=create_model()
    m.load_state_dict(ckpt['state_dict'])
    m.eval()
    return m
def read_image(f):
    data=f.getvalue()
    if len(data)>12*1024*1024: raise ValueError('Maximum image size is 12 MB')
    return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB')
@torch.inference_mode()
def predict(model,image):
    p=torch.softmax(model(T(image).unsqueeze(0)),dim=1).squeeze(0).tolist()
    return sorted(zip(CLASSES,p),key=lambda x:x[1],reverse=True)
