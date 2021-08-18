import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from script15_unit import CustomImageDataset
from script15_unit import Net_resnet, SmoothCrossEntropy
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
import numpy as np
import argparse


np.set_printoptions(linewidth=150)

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

categ_id = {v:k for k,v in id_categ.items()}



def train(model, namesave, device, fullaug, batch):
    training_data = CustomImageDataset(
        train=True, device=device, fullaug=fullaug
    )
    print('len train {}'.format(len(training_data)))

    testing_data = CustomImageDataset(
        train=False, device=device, fullaug=fullaug
    )
    print('len test {}'.format(len(testing_data)))

    train_dataloader = DataLoader(training_data, batch_size=batch, shuffle=True)
    testloader = DataLoader(testing_data, batch_size=batch, shuffle=False)
    n_epoch = 15

    criterion = nn.CrossEntropyLoss()
    # criterion = SmoothCrossEntropy()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    maxval = 0
    print('start train')
    N = len(train_dataloader)

    for epoch in range(n_epoch):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        start = time.time()
        for i, data in enumerate(train_dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss= criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # if i != 0 and i % 200 == 0:
            #     print('{}/{} - {:.3f}'.format(i, N, loss.item()))
        print('epoch {}.  loss: {:.3f}'.format(epoch + 1, running_loss/N))
        model.eval()


        predictions = []
        trues = []
        loss_val = 0
        with torch.no_grad():
            for data in testloader:
                inputs, label = data
                y_pred = model(inputs)
                loss = criterion(y_pred, label)
                loss_val += loss.item()
                predictions += list(torch.argmax(F.softmax(y_pred, dim=1), dim=1).detach().cpu().numpy())
                trues += list(label.detach().cpu().numpy())
        score1 = f1_score(trues, predictions, average='macro', zero_division=0)
        print(classification_report(trues, predictions, zero_division=0, digits=3))
        print(confusion_matrix(trues, predictions))

        if maxval <= score1:
            # PATH = './model/resnet/{}.pth'.format(namesave)
            PATH = './model/resnet/{}_{:.3f}.pth'.format(namesave, score1)
            torch.save(model.state_dict(), PATH)
            maxval = score1

        print('time train {:.1f}'.format(time.time()-start))
        scheduler.step()
        print('end epoch {}\n'.format(epoch+1))
        # running_loss = 0.0
    print('Finished Training')

    # PATH = './model_cv.pth'
    # torch.save(model.state_dict(), PATH)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    # parser.add_argument('--rand', type=int, default=42)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--net', type=str, default='b3')
    # parser.add_argument('--aug', default=False,  action='store_true')
    # parser.add_argument('--froze', default=False,  action='store_true')
    # parser.add_argument('--newclass', default=False,  action='store_true')
    args = parser.parse_args()
    print(args)
    device = args.device
    # device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    resnet = Net_resnet(args.net, False, False).to(device)
    # net .......  aug, froz, newclass
    namef = '{}_1234'.format(args.net)
    print(namef)
    train(resnet, namef,  device, True, args.batch)
