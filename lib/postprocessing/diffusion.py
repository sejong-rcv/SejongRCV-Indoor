import numpy as np
import scipy.sparse as sparse

from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn import preprocessing
from lib import metric as mt
from lib import postprocessing as pp
import scipy.sparse.linalg as linalg

class Diffusion_search():
    def __init__(self, query_feat, index_feat, trancation_size=1000, pca=None, kd=50):
        self.query_feat = query_feat
        self.index_feat = index_feat
        self.trc_size = trancation_size
        self.pca = pca
        self.kd = kd

    def search(self):
        n_query = len(self.query_feat)
        diffusion = Diffusion(np.vstack([self.query_feat, self.index_feat]))
        offline = diffusion.get_offline_results(self.trc_size, self.kd)
        features = preprocessing.normalize(offline, norm="l2", axis=1)
        scores = features[:n_query] @ features[n_query:].T
        ranks = np.argsort(-scores.todense())
        ranks = np.asarray(ranks)

        return ranks


class Diffusion(object):
    """Diffusion class
    """
    def __init__(self, features):
        self.features = features
        self.N = len(self.features)
        # use ANN for large datasets
        self.use_ann = self.N >= 100000
        if self.use_ann:
            self.ann = mt.ANN_faiss(self.features, method='cosine')
        self.knn = mt.KNN_faiss(self.features, method='cosine')

        self.trunc_ids=None
        self.trunc_init=None
        self.lap_alpha=None

    def get_offline_result(self,i):
        ids = self.trunc_ids[i]
        trunc_lap = self.lap_alpha[ids][:, ids]
        scores, _ = linalg.cg(trunc_lap, self.trunc_init, tol=1e-6, maxiter=20)
        return scores

    def get_offline_results(self, n_trunc, kd=50):
        """Get offline diffusion results for each gallery feature
        """
        # print('[offline] starting offline diffusion')
        # print('[offline] 1) prepare Laplacian and initial state')
        
        if self.use_ann:
            _, self.trunc_ids = self.ann.search(self.features, n_trunc)
            sims, ids = self.knn.search(self.features, kd)
            self.lap_alpha = self.get_laplacian(sims, ids)
        else:
            sims, ids = self.knn.search(self.features, n_trunc)
            self.trunc_ids = ids
            self.lap_alpha = self.get_laplacian(sims[:, :kd], ids[:, :kd])
        self.trunc_init = np.zeros(n_trunc)
        self.trunc_init[0] = 1

        # print('[offline] 2) gallery-side diffusion')
        results = Parallel(n_jobs=-1, prefer='threads')(delayed(self.get_offline_result)(i)
                                      for i in tqdm(range(self.N),
                                                    desc='[offline] diffusion'))
        all_scores = np.concatenate(results)
        
        # print('[offline] 3) merge offline results')
        rows = np.repeat(np.arange(self.N), n_trunc)
        offline = sparse.csr_matrix((all_scores, (rows, self.trunc_ids.reshape(-1))),
                                    shape=(self.N, self.N),
                                    dtype=np.float32)
        return offline


    def get_laplacian(self, sims, ids, alpha=0.99):
        """Get Laplacian_alpha matrix
        """
        affinity = self.get_affinity(sims, ids)
        num = affinity.shape[0]
        degrees = affinity @ np.ones(num) + 1e-12
        # mat: degree matrix ^ (-1/2)
        mat = sparse.dia_matrix(
            (degrees ** (-0.5), [0]), shape=(num, num), dtype=np.float32)
        stochastic = mat @ affinity @ mat
        sparse_eye = sparse.dia_matrix(
            (np.ones(num), [0]), shape=(num, num), dtype=np.float32)
        lap_alpha = sparse_eye - alpha * stochastic
        return lap_alpha

    def get_affinity(self, sims, ids, gamma=3):
        """Create affinity matrix for the mutual kNN graph of the whole dataset
        Args:
            sims: similarities of kNN
            ids: indexes of kNN
        Returns:
            affinity: affinity matrix
        """
        num = sims.shape[0]
        sims[sims < 0] = 0  # similarity should be non-negative
        sims = sims ** gamma
        # vec_ids: feature vectors' ids
        # mut_ids: mutual (reciprocal) nearest neighbors' ids
        # mut_sims: similarites between feature vectors and their mutual nearest neighbors
        vec_ids, mut_ids, mut_sims = [], [], []
        for i in range(num):
            # check reciprocity: i is in j's kNN and j is in i's kNN when i != j
            ismutual = np.isin(ids[ids[i]], i).any(axis=1)
            ismutual[0] = False
            if ismutual.any():
                vec_ids.append(i * np.ones(ismutual.sum(), dtype=int))
                mut_ids.append(ids[i, ismutual])
                mut_sims.append(sims[i, ismutual])
        vec_ids, mut_ids, mut_sims = map(np.concatenate, [vec_ids, mut_ids, mut_sims])
        affinity = sparse.csc_matrix((mut_sims, (vec_ids, mut_ids)),
                                     shape=(num, num), dtype=np.float32)
        return affinity