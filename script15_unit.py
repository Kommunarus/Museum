import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models
import os
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
from PIL import Image
from sklearn.model_selection import train_test_split
import albumentations as A
import numpy as np
import pickle as pkl
import geffnet


transform_Albu = A.Compose([
    A.Cutout(),
    A.HorizontalFlip(),
    A.Rotate(limit=(-20, 20), interpolation=1),
    A.ToGray(),
])
transform_full_Albu = A.Compose([
    # A.CLAHE(),
    A.Cutout(),
    A.HorizontalFlip(),
    A.Rotate(limit=(-20, 20), interpolation=1),
    # A.Flip(),
    # A.RandomRotate90(),
    # A.Transpose(),
    A.RandomBrightnessContrast(),
    A.HueSaturationValue(),
    A.ToGray(),
    # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
    # A.Blur(blur_limit=3),
    A.OpticalDistortion(),
    A.GridDistortion(),
])

random.seed(21)
Image.MAX_IMAGE_PIXELS = None
num_class = 18

id_categ = {
	0:         'живопись' ,
	1:         'графика' ,
	2:         'скульптура'  ,
	3:         'изделия прикладного искусства',
	4:         'предметы нумизматики'   ,
	5:         'предметы археологии'    ,
	6:         'предметы этнографии'  ,
	7:         'оружие'             ,
	8:         'документы, редкие книги'   ,
	9:         'предметы естественнонаучной коллекции'  ,
	10:        'предметы техники'    ,
	11:        'прочие'   ,
	12:        'предметы прикладного искусства, быта и этнографии' ,
	13:        'редкие книги'     ,
	14:        'документы'   ,
	15:        'предметы печатной продукции'  ,
	16:        'фотографии и негативы'    ,
	17:        'предметы минералогической коллекции'
	}

categ_id = {v: k for k, v in id_categ.items()}

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
def openfile(x):
    if os.path.exists(os.path.join('/home/alex/PycharmProjects/dataForMC/pkl', f'{x}.pkl')):
        return 1
    else:
        return 0


class CustomImageDataset(Dataset):
    def __init__(self, train, device, fullaug):

        # df4 = pd.read_csv('./data/train_url_only.csv')
        # df4 = df4[pd.notnull(df4['typology'])]
        df1 = pd.read_csv('./data/train_url_xs.csv')
        df1 = df1[pd.notnull(df1['typology'])]
        df2 = pd.read_csv('./data/train_url_xs_2.csv') # zdes net malenkih klassov
        df2 = df2[pd.notnull(df2['typology'])]
        df3 = pd.read_csv('./data/train_url_xs_3.csv')
        df3 = df3[pd.notnull(df3['typology'])]
        df4 = pd.read_csv('./data/train_url_xs_4.csv')
        df4 = df4[pd.notnull(df4['typology'])]
        # df = pd.concat([df1, df2, df3])
        df = pd.concat([df1, df2])
        # df = df1

        df.loc[df["typology"] == "предметы прикладного искусства, быта и этнографии ", "typology"] = \
            'предметы прикладного искусства, быта и этнографии'
        df.loc[df["typology"] == "Фотографии", "typology"] = 'фотографии и негативы'
        df['typology_id'] = df['typology'].apply(lambda x: categ_id[x])
        df['file_ok'] = df['guid'].apply(lambda x: 1 if openfile(x) == 1 else 0)
        df = df[df['file_ok'] == 1]

        _, te = train_test_split(df, train_size=0.9, random_state=1, stratify=df.typology_id)
        test_guid = te['guid'].values

        df = pd.concat([df1, df2, df3, df4])
        df.loc[df["typology"] == "предметы прикладного искусства, быта и этнографии ", "typology"] = \
            'предметы прикладного искусства, быта и этнографии'
        df.loc[df["typology"] == "Фотографии", "typology"] = 'фотографии и негативы'
        df['typology_id'] = df['typology'].apply(lambda x: categ_id[x])
        df['file_ok'] = df['guid'].apply(lambda x: 1 if openfile(x) == 1 else 0)
        df = df[df['file_ok'] == 1]

        tr = df[~df['guid'].isin(test_guid)]

        if train:
            self.dataset = tr
        else:
            self.dataset = te

        self.train = train
        self.fullaug = fullaug


        self.device = device


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        row = self.dataset.iloc[idx]
        guid = row['guid']
        label = row['typology_id']

        with open('/home/alex/PycharmProjects/dataForMC/pkl/{}.pkl'.format(guid), 'rb') as f:
            image = pkl.load(f)
        if self.train:
            if self.fullaug:
                image = transform_full_Albu(image=image)['image']
            else:
                image = transform_Albu(image=image)['image']

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = transforms.ToTensor()(image)
        image = normalize(image)

        return image.to(self.device), torch.tensor(int(label)).to(self.device)





class Net_resnet(nn.Module):
    def __init__(self, str_net, froz, newclass):
        super(Net_resnet, self).__init__()
        if str_net == 'net34':
            model = models.resnet34(pretrained=True)
        elif str_net == 'net18':
            model = models.resnet18(pretrained=True)
        elif str_net == 'net50':
            model = models.resnet50(pretrained=True)
        elif str_net == 'net152':
            model = models.resnet152(pretrained=True)
        elif str_net == 'b2':
            model = geffnet.create_model('efficientnet_b2', pretrained=True)
        elif str_net == 'b3':
            model = geffnet.create_model('efficientnet_b3', pretrained=True)
        elif str_net == 'b4':
            model = geffnet.create_model('tf_efficientnet_b4_ns', pretrained=True)
        elif str_net == 'b7':
            model = geffnet.create_model('tf_efficientnet_b7_ns', pretrained=True)
        elif str_net == 'mixnet':
            model = geffnet.create_model('mixnet_xl', pretrained=True)

        if froz:
            for param in model.parameters():
                param.requires_grad = False

        if str_net in ['b2', 'b3', 'b7', 'mixnet']:
            if newclass:
                model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, 512),
                                                 nn.ReLU(),
                                                 nn.Dropout(p=0.4),
                                                 nn.Linear(512, 128),
                                                 nn.ReLU(),
                                                 nn.Dropout(p=0.4),
                                                 nn.Linear(128, num_class))
            else:
                model.classifier = nn.Linear(model.classifier.in_features, num_class)

            for param in model.classifier.parameters():
                param.requires_grad = True

        if str_net in ['net18', 'net34', 'net50', 'net152']:
            model.fc.out_features = 18
            for param in model.fc.parameters():
                param.requires_grad = True

        self.features = model

    def forward(self, x):
        x = self.features(x)

        return x


class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
            (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()