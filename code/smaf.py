#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 10:26:10 2023

@author: tsuyoshi
"""
# Import libraries
import numpy as np
import anndata as ad
import spams
from utils import sparse_decode_lasso, sparse_decode_blocks_lasso, sparse_decode_omp_fixedk, analyse_U_W

'''
Sparse Module Activity Factorization (SMAF)
(Computes a dictionary (U) from training data)

Find X = UW with special constraints:
inputs:
    SCE: anndata object containing protein expression data (cells x proteins)
    d: maximum (initial) number of modules in the dictionary U (proteins x modules)
    maxItr: number of iteration in SMAF to obtain U,W
    methodW: 'lasso' or 'omp_smallerk' or 'omp_fixedk' for calculating W during SMAF 
        'omp_fixedk': Per cell, find k modules that best correlates with X. 
        'lasso': Per cell, maximise sparsity while constraining error size
            (mode == 1) min_{w} ||w||_1 s.t. ||x-Uw||_2^2 <= (ldaU*||x-Uw||_2^2/Ncells) 
                # Other modes are not implemented yet. 
                # (mode == 0) min_{w} ||x-Uw||_2^2 s.t. ||w||_1 <= lambda1 
                # (mode == 2) min_{w} 0.5||x-Uw||_2^2 + lambda1||w||_1 +0.5 lambda2||w||_2^2
    k: (only used in 'omp_fixedk' ) the number of nonzeros per column in W (default = 3)
    ldaW: (only used in 'lasso') an error threshold coefficient for lasso for W (default = 0.2)
    ldaU: an error threshold coefficient for lasso for U
    THREADS: Number of threads used (default = 20)
    X_normalization: How X is normalized before running smaf (default: L2_norm)
        'L2_norm': protein-wise L2 normalization (Recommended)
        'min_max_norm': protein-wise min_max normalization
        'none': no normalization
    num_blocks_W: (only used in 'lasso') number of blocks used to calculate W. Input 1 for not grouping into blocks. 
    num_blocks_U: number of blocks used to calculate U. Input 1 for not grouping into blocks. Use only when X is not normed
    layer: Name of layer in anndata object to be used as X (default: None, meaning anndata.X)
    Normalize_U: If True, U is module-wise L2 normalized at every iteration of SMAF. (default: True)
    saveItr: If True, data of U,W for all iterations are calculated and output as "results" and "colnames" (default: False)

outputs:
    U: a dictionary of gene modules (proteins x modules) 
    W: a module activity levels in each cell of training data(modules x cells)
    X: the protein expression data used for SMAF (proteins x cells)
    
    Two additional outputs when saveItr == True
    results: U and W properties for each iteration (list of list)
    colnames: colnames for the results (list)

'''

def smaf(SCE,d,maxItr,methodW,ldaU, ldaW=0.2, k=3, THREADS=20, X_normalization='paper_norm',
          num_blocks_W=20, num_blocks_U=1, layer=None, Normalize_U=True, saveItr=False):

    # Select layer of anndata object that should be used in SMAF and transpose it to proteins x cells
    if not isinstance(SCE,ad._core.anndata.AnnData): # alternatively, input can be np array(float64) of proteins x cells
        X_mat = SCE
    elif layer == 'counts':
        X_mat = (SCE.X).T
    elif layer is not None :
        X_mat = (SCE.layers[layer]).T
    else:
        X_mat = (SCE.X).T

    # Extract data matrix X from anndata object and apply selected normalization
    match X_normalization:
        case 'paper_norm':
            X = (X_mat.T / np.linalg.norm(X_mat, axis=1)).T
        case 'min_max_norm':
            X = (X_mat-X_mat.min(axis=1, keepdims=True)) / (
                X_mat.max(axis=1, keepdims=True)-X_mat.min(axis=1, keepdims=True))
        # case 'zscore_norm':
        #     X = ((X_mat.T-np.mean(X_mat, axis=1)) / np.std(X_mat, axis=1)).T # zscore(X_mat, axis=1)
        case 'none':
            X = X_mat
        case _:
            # In case no valid normalization is given, an error is thrown
            raise ValueError(('The normalization {0} used by smaf is not valid.'.format(normalization) +
                              'Please use one of the following: paper_norm, min_max_norm, or none.'))
 
    # Initialze U, W from NMF
    U, W = spams.nmf(np.asfortranarray(X), return_lasso=True, K=d, numThreads=THREADS)
    W = np.asarray(W.todense())
    print('Initialized U, W with NMF, SMAF maxItr = ',maxItr)
    
    # prepare output results
    results = []

    # For maxItr iterations approximate U, W by:
    # a. Update the module dictionary as U=LassoNonNegative()
    # b. Normalize each module with L2-norm=1
    # c. Update the activity levels as W=OMP()or LassoNonNegative() 
    for itr in range(maxItr):
        # Calc. U
        # Higher ldaU will be a worse fit, but will result in a sparser solution
        if num_blocks_U == 1:
            Ut = sparse_decode_lasso(X.T, W.T, ldaU, THREADS)
            U = Ut.T
        else:
            Ut = sparse_decode_blocks_lasso(X.T, W.T, ldaU, THREADS, num_blocks_U)
            U = Ut.T
        # Remove empty columns (modules containing no proteins) 
        U = U[:, (U.sum(0) > 0)]
        # Normalize U module-wise. Default: True
        if Normalize_U:
            U = U / np.linalg.norm(U, axis=0)
            U[np.isnan(U)] = 0
            
        # Calc. W
        match methodW:
            case 'omp_fixedk':
                W = sparse_decode_omp_fixedk(X, U, k, THREADS)
                
            case 'lasso':
                if num_blocks_W == 1:                    
                    W = sparse_decode_lasso(X, U, ldaW, THREADS)
                else:                    
                    W = sparse_decode_blocks_lasso(X, U, ldaW, THREADS, num_blocks_W)
        
        # Save UW data for each iteration
        if saveItr:
            # print('itr =', itr)
            cur_results, colnames = analyse_U_W(U, W, X)
            results.append(cur_results)
        
    if np.shape(U)[1] == 0:
            # In case U is empty throw error, since CISI didn't work
            raise ValueError(('CISI failed when computing U (dictionary). ' +
                              'The computed U has zero columns.'))
    if saveItr:
        return U, W, X, results, colnames

    return U, W, X

