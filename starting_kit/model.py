import numpy as np 
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import tree
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

class preprocess:
	def __init__(self, n_components=10):
		self.n_components = n_components
		self.pca = None
	
	def dimensionReduction_fit(self, data):
		self.pca = PCA(n_components=self.n_components)
		realData = np.ndarray(shape=(data.shape[0], self.n_components))
		self.pca.fit(data)

	def dimensionReduction_transform(self, data):
		realData = self.pca.transform(data)
		return realData


	def fit(self, data):
		#entrainement du processing
		self.dimensionReduction_fit(data)

		#c'est ici qu'il faut rajouter l'appel a d'autre preprocess (si besoin d'entrainement)
		#sinon directement dans transform

	def transform(self, data):
		#transformation des données a l'aide d'un preprocessing deja entrainé
		data_transformed = self.dimensionReduction_transform(data)

		#ici on rappel pour d'autre preprocess si pas besoin d'entrainement

		return data_transformed


	def fit_transform(self, data):
		self.fit(data)
		data_transformed = self.transform(data)
		return data_transformed


class model (BaseEstimator):
    def __init__(self):
        self.num_train_samples=0
        self.is_trained=False
        self.preprocess = preprocess(n_components=9)
        # 5 -> 0.5982865876
        # 8 -> 0.8986549364
        # 9 -> 0.9014957412
        # 10 -> 0.9009779967
        # 11 -> 0.8969211935
        # 15 -> 0.8955047299
        # 59 -> 0.8916412716
        # plus on s'approche de 9, plus le score est bon
        self.mod = tree.DecisionTreeRegressor()
    
    def fit(self, X, y):

    	#preprocessing
        X = self.preprocess.fit_transform(X)

        #entrainement
        self.mod.fit(X,y)
        self.is_trained = True

    def predict(self, X):
        X = self.preprocess.transform(X) #preprocessing
        y = self.mod.predict(X) #prediction
        return y

    def save(self, outname='model'):
        pass
        
    def load(self):
        return self