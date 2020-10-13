import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import csv

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
from sklearn.metrics import fbeta_score
from sklearn import metrics
# from efficientnet_pytorch import EfficientNet

input_path = "./DATASET_FINAL/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Batch Size
batch_size=24



## Generators
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(60, resample=False, expand=False, center=None, fill=None),
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

image_datasets = {
    'train': 
    datasets.ImageFolder(input_path + 'train', data_transforms['train']),
    'validation': 
    datasets.ImageFolder(input_path + 'validation', data_transforms['validation'])
}

# print(image_datasets["train"].class_to_idx)

dataloaders = {
    'train':
    torch.utils.data.DataLoader(image_datasets['train'],
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=2),  
    'validation':
    torch.utils.data.DataLoader(image_datasets['validation'],
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=2)  
}


## efficientnet
# model = EfficientNet.from_pretrained('efficientnet-b7')
# # Unfreeze model weights
# for param in model.parameters():
#     param.requires_grad = True
# num_ftrs = model._fc.in_features
# model._fc = nn.Linear(num_ftrs, 3)
# model = model.to('cuda')

# optimizer = optim.Adam(model.parameters())
# loss_func = nn.BCELoss()
# criterion = loss_func 


# ### RESNET
# model = models.wide_resnet50_2(pretrained=True).to(device)
model = models.wide_resnet50_2(pretrained=True).to(device)
# model=MyCustomResnet18(nn.Module)
for param in model.parameters():
    param.requires_grad = False   
    
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 3)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())


### Train
def train_model(model, criterion, optimizer, num_epochs=3):
    torch.multiprocessing.freeze_support()
    f=open('result.csv', mode='w')
    writer=csv.writer(f, delimiter=',')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                # ## F2 Score
                # logits=outputs.cpu().detach().numpy()
                # logits2=[]
                # for i in range(logits.shape[0]):
                #     pred=np.argmax(logits[i])
                #     logits2.append(pred)
                # label=labels.cpu().detach().numpy()

                # y_true=label
                # y_pred=logits2
                # F2_score=fbeta_score(y_true, y_pred, average='macro', beta=0.5)


                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)

                ## F2 Score
                logits=preds.cpu().detach().numpy()
                label=labels.cpu().detach().numpy()
                y_true=label
                y_pred=logits
                F2_score=fbeta_score(y_true, y_pred, average='macro', beta=2.0)

                # AUC
                fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=2)
                auc=metrics.auc(fpr, tpr)


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            if phase=='train':
                print('{} loss: {:.4f}, acc: {:.4f}, f2_score: {:.4f}, auc_score: {:.4f}'.format(phase,
                                                            epoch_loss,
                                                            epoch_acc,F2_score,auc))
                torch.save(model.state_dict(), 'models/pytorch/weights.h5')
            else:
                print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                            epoch_loss,
                                                            epoch_acc))

            writer.writerow([phase, epoch_loss, epoch_acc,F2_score,auc])
    f.close()
    return model

if __name__ == '__main__':
    ## Training
    model_trained = train_model(model, criterion, optimizer, num_epochs=100)

    ### Save Model
    torch.save(model_trained.state_dict(), 'models/pytorch/weights.h5')

# ## Load Model
# model = models.resnet50(pretrained=False).to(device)
# model.fc = nn.Sequential(
#                nn.Linear(2048, 128),
#                nn.ReLU(inplace=True),
#                nn.Linear(128, 3)).to(device)
# model.load_state_dict(torch.load('models/pytorch/weights.h5'))



# ## make predictions
# validation_img_paths = ["validation/COVID/93FE0BB1-022D-4F24-9727-987A07975FFB.jpeg",
#                         "validation/NORMAL/IM-0050-0001.jpeg",
#                         "validation/PNEUMONIA/person11_virus_38.jpeg"]
# img_list = [Image.open(input_path + img_path) for img_path in validation_img_paths]

# validation_batch = torch.stack([data_transforms['validation'](img).to(device)
#                                 for img in img_list])

# pred_logits_tensor = model(validation_batch)
# pred_logits_tensor

# pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
# pred_probs

# ## Plot
# fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
# for i, img in enumerate(img_list):
#     ax = axs[i]
#     ax.axis('off')
#     ax.set_title("{:.0f}% COVID, {:.0f}% NORMAL,{:.0f}% PNEUMONIA".format(100*pred_probs[i,0],
#                                                             100*pred_probs[i,1],
#                                                             100*pred_probs[i,2]))
#     ax.imshow(img)

# plt.show()
