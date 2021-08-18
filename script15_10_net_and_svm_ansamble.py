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

def getinput(guid, donormaliz):
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
		if donormaliz:
			image = normalize(image)
		return image.to(device)
	except:
		return None

def most_frequent(List):
    return max(set(List), key = List.count)


device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

net_ansamble = []
# net_ansamble.append(('b3',      './ansamble/b3_12_T_F_F_0.741.pth', True)) # old best solo 0.7599987614269705
# net_ansamble.append(('net34',   './ansamble/cv_best_resnet_0.772.pth', False)) # old best solo 0.761339107544574
# # new
# net_ansamble.append(('b3',      './ansamble/b3_123_0.749.pth', True)) # 0.79013
# #new new
# net_ansamble.append(('b3',      './ansamble/b3_1234_0.753.pth', True)) # 0.78386
# #
# net_ansamble.append(('b3',      './ansamble/b3_1234_0.755.pth', True)) # 0.78

# net_ansamble = []
for ves in os.listdir('./model/resnet'):
	if ves[:2] == 'b3':
		net_ansamble.append(('b3', './model/resnet/'+ves, True))
	# elif ves[:5] == 'net18':
	# 	net_ansamble.append(('net18', './model/resnet/'+ves, True))
	# elif ves[:5] == 'net34':
	# 	net_ansamble.append(('net34', './model/resnet/'+ves, True))
	# elif ves[:6] == 'mixnet':
	# 	net_ansamble.append(('mixnet', './model/resnet/'+ves, True))
	elif ves == 'cv_best_resnet_0.772':
		net_ansamble.append(('net34', './model/resnet/'+ves, False))
	# elif ves[:16] == 'cv_best_resnet34':
	# 	net_ansamble.append(('net34', './model/resnet/'+ves, True))
	# elif ves[:17] == 'cv_best_resnet152':
	# 	net_ansamble.append(('net152', './model/resnet/'+ves, True))



dftest = pd.read_csv('./data/test.csv')
dftest = dftest.fillna('')

total_ans = []
for j in range(len(net_ansamble)):
	nett =  net_ansamble[j][0]
	model = Net_resnet(nett, False, False).to(device)
	model.load_state_dict(torch.load( net_ansamble[j][1]))
	model.eval()
	totall = []
	for i, row in dftest.iterrows():
		with torch.no_grad():
			inputs = getinput(row.guid, net_ansamble[j][2])
			if inputs != None:
				y_pred = model(inputs)
				totall += list(torch.argmax(F.softmax(y_pred, dim=1), dim=1).detach().cpu().numpy())
			else:
				totall += ['nan']
	total_ans.append(totall)

total = []
for aa in zip(*total_ans):
	total.append(most_frequent(list(aa)))
dftest['typology_net'] = total
print('svm')
dftest['typology_svm'] = predict_svm(False)
# dftest['typology'] = 'empty'
dftest['typology'] = 'предметы прикладного искусства, быта и этнографии'
print('go')

# for i, row in dftest.iterrows():
# 	inputs = getinput(row.guid)
# 	if inputs != None:
# 		if row['typology_net'] != 'nan':
# 			row['typology'] = id_categ[row['typology_net']]
# 	elif row['description'] != '':
# 		row['typology'] = row['typology_svm']

for i, row in dftest.iterrows():
	if row['description'] != '':
		row['typology'] = row['typology_svm']
	else:
		if row['typology_net'] != 'nan':
			row['typology'] = id_categ[row['typology_net']]



dftest = dftest[['guid', 'typology']]
dftest.to_csv(f'ansamble_14.csv', index=False, header=['guid', 'typology'])
