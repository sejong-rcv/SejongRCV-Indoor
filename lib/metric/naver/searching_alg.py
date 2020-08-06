import faiss
import numpy as np
import os
from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors

def kNN(index, predict, k=1):
    kNN = NearestNeighbors(n_neighbors=k)
    kNN.fit(index)
    topk_list = kNN.kneighbors(predict, return_distance=False)
    return topk_list

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


class BaseKNN(object):
    """KNN base class"""
    def __init__(self, database, method):
        if database.dtype != np.float32:
            database = database.astype(np.float32)
        self.N = len(database)
        self.D = database[0].shape[-1]
        self.database = database if database.flags['C_CONTIGUOUS'] \
                               else np.ascontiguousarray(database)

    def add(self, batch_size=10000):
        """Add data into index"""
        if self.N <= batch_size:
            self.index.add(self.database)
        else:
            [self.index.add(self.database[i:i+batch_size])
                    for i in tqdm(range(0, len(self.database), batch_size),
                                  desc='[index] add')]

    def search(self, queries, k):
        """Search
        Args:
            queries: query vectors
            k: get top-k results
        Returns:
            sims: similarities of k-NN
            ids: indexes of k-NN
        """
        if not queries.flags['C_CONTIGUOUS']:
            queries = np.ascontiguousarray(queries)
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)
        sims, ids = self.index.search(queries, k)
        return sims, ids


class KNN_faiss(BaseKNN):
    """KNN class
    Args:
        database: feature vectors in database
        method: distance metric
    """
    def __init__(self, database, method):
        super().__init__(database, method)
        self.index = {'cosine': faiss.IndexFlatIP,
                      'euclidean': faiss.IndexFlatL2}[method](self.D)
        if os.environ.get('CUDA_VISIBLE_DEVICES'):
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        self.add()


class ANN_faiss(BaseKNN):
    """Approximate nearest neighbor search class
    Args:
        database: feature vectors in database
        method: distance metric
    """
    def __init__(self, database, method, M=128, nbits=8, nlist=316, nprobe=64):
        super().__init__(database, method)
        self.quantizer = {'cosine': faiss.IndexFlatIP,
                          'euclidean': faiss.IndexFlatL2}[method](self.D)
        self.index = faiss.IndexIVFPQ(self.quantizer, self.D, nlist, M, nbits)
        samples = database[np.random.permutation(np.arange(self.N))[:self.N // 5]]
        print("[ANN] train")
        self.index.train(samples)
        self.add()
        self.index.nprobe = nprobe
