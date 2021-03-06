from hmmlearn import hmm
import numpy as np
import pickle

"""
class GMM_HMM model
"""

class GMM_HMM():
    def __init__(self, n_components=5, cov_type='diag', n_iter=100):
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter

        self.model = hmm.GMMHMM(n_components=self.n_components,
                                covariance_type=self.cov_type, n_iter=self.n_iter)

    def train(self, X):
        np.seterr(all='ignore')
        self.model.fit(X)

    def get_score(self, input_data):
        return self.model.score(input_data)


