import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms as T
from PIL import Image
import io
import requests
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# define model
def Net():
    model = models.__dict__['resnet50'](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    num_features = model.fc.in_features
    
    model.fc = nn.Sequential(
    nn.Linear(num_features, int(num_features/2)),
    nn.Linear(int(num_features/2), 5))
    return model

# load model parameters
def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    model.eval()
    return model
    
# deserialize input
def input_fn(request_body, content_type):
    if content_type == 'image/jpeg':
        img = Image.open(io.BytesIO(request_body))
        return img
    else:
        raise ValueError("This model only supports jpeg input")

# inference
def predict_fn(input_object, model):
    transform = T.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])
    input_object=transform(input_object)
    input_object=input_object.unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(input_object)
    return prediction

'''
# postprocess
def output_fn(predictions, content_type):
    assert content_type == "application/json"
    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)

# Serialize the prediction result into the desired response content type
def output_fn(prediction, accept=JSON_CONTENT_TYPE):        
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE: 
        logger.debug(f'Returning response {json.dumps(prediction)}')
        return json.dumps(prediction), accept
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))
'''