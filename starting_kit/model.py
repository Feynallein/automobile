import numpy as np 
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

class preprocess:
	def __init__(self, n_components=10):
		self.n_components = n_components
		self.pca = None

	#def outlierDetection(self, data):
	    #clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
	    #outlier = clf.fit_predict(data)
	    #countInlier = 0
	    #for i in outlier:
	    #    if i == 1:
	    #        countInlier = countInlier + 1
	    #realData = np.ndarray(shape=(countInlier,60))
	    #count = 0
	    #for i in range(len(outlier)):
	    #    if outlier[i] == 1:
	    #        realData[count] = data[i]
	    #        count = count + 1
	    #return realData

	#def featuresSelection(slef, data, dataTarget, optionalEstimator, max_depth_tree, min_sample_leaf_tree):
	 	#clf = ExtraTreesClassifier(n_estimators=optionalEstimator, max_depth=max_depth_tree, min_samples_leaf=min_sample_leaf_tree) 
	  	#clf = clf.fit(data, dataTarget)
	    #model = SelectFromModel(clf, threshold="mean", prefit=True)
	    #realData = model.transform(data)
	    #return realData
	
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
        self.preprocess = preprocess(n_components=10)
        self.mod = linear_model.Ridge()
    
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