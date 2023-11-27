#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 13:53:07 2023

@author: tsuyoshi
"""

import numpy as np
import os
from utils import random_phi_subsets_g, check_balance, get_observations, sparse_decode_blocks_lasso, correlations
from scipy.spatial import distance   

def simulate_A(X_input, U, phi, inverted_snr, decoding_lasso_lda = 0.02,
                     outpath=None, suffix = "", THREADS=20, layer=None, num_blocks=20):
    
    # Select layer of anndata object that should be used in SMAF and transpose it to proteins x cells    
    if layer == 'counts':
        X = (X_input.X).T
    elif layer is not None :
        X = (X_input.layers[layer]).T
    else:
        X = (X_input.X).T
    # Simulate composite oservations Y using X and noise
    Y = get_observations(X, phi, inverted_snr=inverted_snr)
    # Compute W given Y, A and U
    W = sparse_decode_blocks_lasso(Y, phi.dot(U), decoding_lasso_lda, 
                                   numThreads=THREADS, num_blocks=num_blocks)
    Xhat = U.dot(W)
    Xhat[np.isnan(Xhat)] = 0
    if outpath is not None:        
        # np.save(os.path.join(outpath, 'X_decoded_sim.npy'),Xhat)
        # np.save(os.path.join(outpath, 'X_groundtruth.npy'),X)
        Xhat_anndata = X_input.copy()
        for k in list(Xhat_anndata.layers.keys()):
            del Xhat_anndata.layers[k]
        Xhat_anndata.X = Xhat.T
        Xhat_anndata.write(os.path.join(outpath, 'X_decoded_sim_{}.h5ad'.format(suffix)))
        X_save = X_input.copy()
        X_save.X = X.T
        for l in list(X_save.layers.keys()):
            del X_save.layers[l]
        X_save.write(os.path.join(outpath, 'X_groundtruth_{}.h5ad'.format(suffix)))

    return X, Xhat, W, Y
