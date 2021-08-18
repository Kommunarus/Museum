import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pickle

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


def most_frequent(List):
    return max(set(List), key = List.count)

def predict_svm(fitting = False):
    col = ['typology', 'description']
    print('read')
    filename = 'finalized_model_svm.sav'
    filenamevec = 'vectorizer.pk'

    if fitting:

        df = pd.read_csv('./data/big_data/train.csv', names=['description', 'typology', 'typology_2', 'typology_3'],
                         delimiter=';', dtype='unicode', header=0)
        df = df[col]
        df = df[pd.notnull(df['description'])]
        df = df[pd.notnull(df['typology'])]
        df.loc[df["typology"] == "Фотографии", "typology"] = 'фотографии и негативы'
        df.loc[df["typology"] == "предметы прикладного искусства, быта и этнографии ", "typology"] \
            = 'предметы прикладного искусства, быта и этнографии'

        df['typology_id'] = df['typology'].apply(lambda x: categ_id[x])

        # df['typology_id'] = df['typology'].factorize()[0]
        # category_id_df = df[['typology', 'typology_id']].drop_duplicates().sort_values('typology_id')
        # # category_to_id = dict(category_id_df.values)
        # id_to_category = dict(category_id_df[['typology_id', 'typology']].values)

        # tfidf = TfidfVectorizer(sublinear_tf=True, min_df=3, norm='l2', ngram_range=(1, 2))
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, norm='l2', ngram_range=(1,3))
        features = tfidf.fit_transform(df.description)
        labels = df.typology_id

        with open(filenamevec, 'wb') as fin:
            pickle.dump(tfidf, fin)

        print('fit')

        model1 = LinearSVC()
        model1.fit(features, labels)
        pickle.dump(model1, open(filename, 'wb'))
    else:
        model1 = pickle.load(open(filename, 'rb'))
        tfidf = pickle.load(open(filenamevec, 'rb'))

    dftest = pd.read_csv('./data/test.csv')
    dftest = dftest.fillna('')
    features_test = tfidf.transform(dftest.description)
    y_pred1 = model1.predict(features_test)
    dftest['typology'] = y_pred1
    dftest['typology2'] = dftest['typology'].apply(lambda row: id_categ[int(row)])
    dftest.loc[dftest["description"] == "", "typology2"] = '.'

    return dftest['typology2']

    # dftest = dftest[['guid','typology2']]
    # dftest.to_csv('ss2.csv', index=False, header=['guid','typology'])
