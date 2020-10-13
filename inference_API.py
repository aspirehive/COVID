import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
from argparse import ArgumentParser
import os.path
import json

input_path = "/home/COVID_RESNET/DATASET_FINAL/"

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

## Generators
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'validation':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ]),
}



### network
model = models.wide_resnet50_2(pretrained=False).to(device)
    
for param in model.parameters():
    param.requires_grad = False   
    
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 3)).to(device)
model.load_state_dict(torch.load("/home/COVID_RESNET/models/pytorch/covid_resnet_wide_trained.h5",map_location=torch.device('cpu')), strict=False)
model.eval()


if __name__ == "__main__":
    # execute only if run as a script
    parser = ArgumentParser(description="ikjMatrix multiplication")
    parser.add_argument("-i", dest="filename", required=True,
                        help="Input File Path", metavar="FILE")
    args = parser.parse_args()

    arg=args.filename
    path_type=['png','jpg','jpeg','tif']
    if arg.split('.')[-1] not in path_type:
        parser.error("The file %s is not a valid image file!" % arg)


    # make predictions
    validation_img_paths = [arg]
    img_list = [Image.open(img_path).convert('RGB') for img_path in validation_img_paths]

    validation_batch = torch.stack([data_transforms['validation'](img).to(device)
                                    for img in img_list])


    pred_logits_tensor = model(validation_batch)
    # print(pred_logits_tensor)

    pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
    # # print(pred_probs)
    # print("{:.2f}% COVID, {:.2f}% NORMAL,{:.2f}% PNEUMONIA".format(100*pred_probs[0,0],
    #                                                         100*pred_probs[0,1],
    #                                                         100*pred_probs[0,2]))
    ##JSON
    jss={"COVID":str(100*pred_probs[0,0]), 
          "NORMAL":str(100*pred_probs[0,1]), 
          "PNEUMONIA":str(100*pred_probs[0,2])}
    print(jss)
