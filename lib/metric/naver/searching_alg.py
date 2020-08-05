from sklearn.neighbors import NearestNeighbors
import faiss
import numpy as np
def kNN(index, predict, k=1):
    kNN = NearestNeighbors(n_neighbors=k)
    kNN.fit(index)
    topk_list = kNN.kneighbors(predict, return_distance=False)
    return topk_list

def kNN_GPU_(index, predict, k=1):
    print("Start KNN GPU")
    #import pdb;pdb.set_trace()
    index=np.array(index)
    predict=np.array(predict)
    index=index.astype("float32")
    predict=predict.astype("float32")
    d=index.shape[1]
    kNN = kNN_Gpu(d=d,GPU=True)

    kNN.train(index)
    D,topk_list = kNN.predict(predict, k)
    return D,topk_list
# import faiss                   # make faiss available

class kNN_Gpu():
    def __init__(self, d=64, GPU=False, GPU_Number=0): #default dimension=64
        self.idx = faiss.IndexFlatL2( d )   # build the index
        if GPU:
            self.res = faiss.StandardGpuResources()  # use a single GPU
            gpu = faiss.index_cpu_to_gpu(self.res, GPU_Number, self.idx)
            self.idx = gpu

    def train(self, index):
        if self.idx.is_trained:
            self.idx.add(index)
            print("number of index : {}".format(self.idx.ntotal))
            return True
        else:
            return False
        
    def predict(self, query, k=1):
        D, I = self.idx.search(query, k)     # actual search
        return D,I
    
    def train_predict(self, index, query, k=1):
        self.idx.is_trained
        self.idx.add(index)
        
        #print("total knn" + index.ntotal) 
        D, I = self.idx.search(query, k)     # actual search

        return I

class kNN_GPU():
    def __init__(self, d=64, GPU=False, GPU_Number=0): #default dimension=64
        self.idx = faiss.IndexFlatL2( d )   # build the index
        self.GPU = GPU
        if self.GPU:
            self.res = faiss.StandardGpuResources()  # use a single GPU
            gpu = faiss.index_cpu_to_gpu(self.res, GPU_Number, self.idx)
            self.idx = gpu
    def train(self, index):
        if self.idx.is_trained:
            self.idx.add(index)
            return
        else:
            raise ValueError('kNN GPU Error')
    def predict(self, query, k=1):
        D, I = self.idx.search(query, k)     # actual search
        return I
    
    def delete(self):
        del self.idx

        if self.GPU:
            del self.res
        
        return
    