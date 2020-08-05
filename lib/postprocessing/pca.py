
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import pickle as pk


    
class PCAwhitening():
    def __init__(self, pca_dim=4096, pca_whitening=True):
        self.pca_dim = pca_dim
        self.pca_whitening = pca_whitening
        self.pca=PCA(n_components=self.pca_dim, whiten=self.pca_whitening)
    def __L2norm__(self, feats):
        feats_norm = normalize(feats, axis=1)
        return feats_norm

    def fit(self, features):
        # Like Delf postprocessing
        # L2 -> pca -> L2
        #feats_norm = self.__L2norm__(features)

        
        output = self.pca.fit(features)

        #output = self.__L2norm__(output)
        
        #return output
    def transform(self, features):
        # Like Delf postprocessing
        # L2 -> pca -> L2
        #feats_norm = self.__L2norm__(features)
        output = self.pca.transform(features)

        #output = self.__L2norm__(output)
        
        #return output
    def save(self,floor,name):
        pk.dump(self.pca, open("pca_%s_%s.pkl"% (floor,name),"wb"))
    def load(self,floor,name):
        self.pca=pk.load( open("pca_%s_%s.pkl"% (floor,name),"rb"))

    


