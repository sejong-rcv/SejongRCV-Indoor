import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

class PCAwhitening():
    def __init__(self, pca_dim=4096, pca_whitening=True):
        self.pca_dim = pca_dim
        self.pca_whitening = pca_whitening
        self.pca = PCA(n_components=self.pca_dim, whiten=self.pca_whitening)
 
        self.whitenp=0.5
        self.whitenv=self.pca_dim
        self.whitenm=1.0

    def __L2norm__(self, feats):
        feats_norm = normalize(feats, axis=1)
        return feats_norm

    def fit_transform(self, features):

        feats_norm = self.__L2norm__(features)
        self.pca.fit(feats_norm)
        output = self.apply(feats_norm)
        output = self.__L2norm__(output)
        
        return output

    def transform(self, features):

        feats_norm = self.__L2norm__(features)
        output = self.apply(feats_norm)
        output = self.__L2norm__(output)
        
        return output

    def apply(self, X):

        if self.pca.mean_ is not None:
            X = X - self.pca.mean_
        X_transformed = np.dot(X, self.pca.components_[:self.whitenv].T)
        if self.pca.whiten:
            X_transformed /= self.whitenm * np.power(self.pca.explained_variance_[:self.whitenv], self.whitenp)

        return X_transformed

        


