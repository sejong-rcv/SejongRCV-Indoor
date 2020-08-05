import pickle
from sklearn.cluster import KMeans, MiniBatchKMeans
from lib.handcraft_extractor.SIFT.SIFT import *
import tqdm

class VLAD(object):
    def __init__(self, ld = "sift"):
        if ld == "root_sift":
            self.local_descriptor = SIFT(dense=True, root=True)
        elif ld == "sift":
            self.local_descriptor = SIFT(dense=True, root=False)
        self.descs_ld = []
        
        self.descs_vlad = []
        self.pose = []
        self.index = []

        self.qdescs_vlad = []
        self.qpose = []
        self.qindex = []
 
    def extract_ld(self, data):
        for i, (img, pose, ind) in enumerate(zip(data['image'], data['pose'], data['index'])):
            (kps, desc) = self.local_descriptor(img)
            self.descs_ld.append(desc.tolist())
            self.pose.append(pose.tolist())
            self.index.append(ind.tolist())


    def build_voca(self, k=256):
        self.descs_ld = np.asarray(self.descs_ld)
        #est = KMeans(n_clusters=k,init='k-means++',tol=0.0001,random_state=0)
        est = MiniBatchKMeans(n_clusters=k,init='k-means++',tol=0.0001,random_state=0)
        _fit_data = np.reshape(self.descs_ld,(-1, 128))
        self.dictionary = est.fit(_fit_data)


    def extract_vlad(self):
 
        num = self.descs_ld.shape[0]
        self.descs_ld = self.descs_ld.tolist()
        for ind in tqdm.tqdm(range(num), desc="extract_vlad"):
            desc = np.asarray(self.descs_ld[0])
            hist = self.dictionary.predict(desc)
            centers = self.dictionary.cluster_centers_
            labels = self.dictionary.labels_
            k = self.dictionary.n_clusters
            m,d = desc.shape
            V=np.zeros([k,d])

            for i in range(k):
                if np.sum(hist==i)>0:
                    V[i]=np.sum(desc[hist==i,:]-centers[i],axis=0)

            V = V.flatten()
            V = np.sign(V)*np.sqrt(np.abs(V))
            V = V/np.sqrt(np.dot(V,V))
            self.descs_vlad.append(V.tolist())
            self.descs_ld.pop(0)
        self.descs_vlad = np.asarray(self.descs_vlad)

            
    
    def extract_vlad_query(self, data):

        for ind, (img, pose, ind) in enumerate(zip(data['image'], data['pose'], data['index'])):
            (kps, desc) = self.local_descriptor(img)
            hist = self.dictionary.predict(desc)
            centers = self.dictionary.cluster_centers_
            labels = self.dictionary.labels_
            k = self.dictionary.n_clusters

            m,d = desc.shape
            V=np.zeros([k,d])

            for i in range(k):
                if np.sum(hist==i)>0:
                    V[i]=np.sum(desc[hist==i,:]-centers[i],axis=0)
            
            V = V.flatten()
            V = np.sign(V)*np.sqrt(np.abs(V))
            V = V/np.sqrt(np.dot(V,V))
            self.qdescs_vlad.append(V.tolist())
            self.qpose.append(pose.tolist())
            self.qindex.append(ind.tolist())

    def get_data(self):

        self.pose = np.asarray(self.pose)
        self.qdescs_vlad = np.asarray(self.qdescs_vlad)
        self.qpose = np.asarray(self.qpose)

        index = {"feat" : self.descs_vlad,
                "pose" : self.pose,
                "index" : self.index}
        
        query = {"feat" : self.qdescs_vlad,
                "pose" : self.qpose,
                "index" : self.qindex}
        
        return index, query

    def save(self, path):
        f = open(path,'wb')
        pickle.dump(self.__dict__, f,  protocol=4)
        f.close()
 
    def load(self, path):
        f = open(path, "rb")
        saved_data = pickle.load(f)
        for key, value in saved_data.items():
            self.__dict__[key] = value
        f.close()
