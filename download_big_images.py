import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
import time
import csv
import json
from io import StringIO
import uuid
import random
from sklearn.utils import resample

df = pd.read_csv('./data/train_url.csv')
# print(len(df))
# df = pd.read_csv('./data/train_url_xs_3.csv')

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

lidf = []
#
for k, v in id_categ.items():
	dfl = df[df['typology'] == v]
	if len(dfl) > 30001:
		indx = random.sample(range(len(dfl)), k=30001)
		lidf.append(dfl.iloc[indx])
	else:
		lidf.append(dfl)

fgh = pd.concat(lidf)
fgh.to_csv('./data/train_url_xs_4.csv', index=False)


# paths = df['guid'].values.tolist()
# urls = df['url'].values.tolist()
#
# def download(url, path1):
# 	time.sleep(random.randint(0,10))
# 	response = requests.get(url)
# 	path2 = f'/home/alex/PycharmProjects/dataForMC/images_3/{path1}.jpg'
# 	with open(path2, 'wb') as f:
# 		f.write(response.content)
#
#
# with ThreadPoolExecutor(max_workers=10) as executor:
#     executor.map(download, urls, paths) #urls=[list of url]



# with open('./data/data-4-structure-3.csv', 'r') as f,  open('./data/train_url.csv', 'w') as fw:
#     reader = csv.reader(f)
#     next(reader)
#     wr = csv.writer(fw)
#     wr.writerow(['guid','description', 'typology', 'url'])
#     for row in reader:
#         susu = row[14]
#         if susu != '':
#             a = StringIO(susu)
#             b = json.load(a)
#             wr.writerow([str(uuid.uuid4()), row[0], row[39], b[0]['url']])
