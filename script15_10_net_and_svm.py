import torch
import pandas as pd
import os
import cv2
from script15_unit import Net_resnet
import torchvision.transforms as transforms

import torch.nn.functional as F
from script1 import predict_svm

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

def getinput(guid):
	path = '/home/alex/PycharmProjects/dataForMC/images'
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])
	try:
		image_raw = cv2.imread(os.path.join(path, f'{guid}.jpg'))
		image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
		# image = Image.open(os.path.join(path, f'{guid}.jpg'))
		# image = image.resize((224, 224), Image.ANTIALIAS)
		image = transforms.ToTensor()(image).unsqueeze(0)
		image = normalize(image)
		return image.to(device)
	except:
		return None


device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

print('net')
nett = 'b3'
model = Net_resnet(nett, False, False).to(device)

model.load_state_dict(torch.load(f'./model/resnet/{nett}_1234_0.755.pth'))
model.eval()

dftest = pd.read_csv('./data/test.csv')
dftest = dftest.fillna('')

total = []
for i, row in dftest.iterrows():
	with torch.no_grad():
		inputs = getinput(row.guid)
		if inputs != None:
			y_pred = model(inputs)
			total += list(torch.argmax(F.softmax(y_pred, dim=1), dim=1).detach().cpu().numpy())
		else:
			total += ['nan']

dftest['typology_net'] = total
print('svm')
dftest['typology_svm'] = predict_svm(False)
dftest['typology'] = 'предметы прикладного искусства, быта и этнографии'
print('go')


for i, row in dftest.iterrows():
	if row['description'] != '':
		row['typology'] = row['typology_svm']
	else:
		if row['typology_net'] != 'nan':
			row['typology'] = id_categ[row['typology_net']]


dftest = dftest[['guid', 'typology']]
dftest.to_csv(f'cv_1234_{nett}_0.755.csv', index=False, header=['guid', 'typology'])
