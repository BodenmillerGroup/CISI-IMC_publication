#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 02:34:39 2023

@author: tsuyoshi
"""
'''
checks that file_path is an existing file and has a certain extension
'''
from pathlib import Path

def is_valid_file(file_path, extension):
    file = Path(file_path)
    if file.exists() and file.is_file() and file.suffix in extension:
        return True
    else:
        return False




'''
Sparse decoding 1
Omp (Orthogonal Matching Persuit) algrithm
'''
import spams
import numpy as np

# # Simplified Omp algorithm. Just tries omp with given k (sparsity).

# Input (Ver compressed): 
#     Y: Compressed measurement (np.array: Compressed channels x Cells)
#     D: Compressed dictionary of gene modules (np.array: Compressed channels x Modules)
#     k: Sparsity of every column in W ()
# Input (Ver not-compressed): 
#     Y: Original measurement (Proteins x Cells)
#     D: Dictionary of gene modules (Proteins x Modules)
#     k: Sparsity of every column in W 
# Output: 
#     W: Module expression (Modules x Cells)

# return W such that Y = DW, and each column of W is k-sparse.

def sparse_decode_omp_fixedk(Y, D, k, numThreads):
    W = spams.omp(np.asfortranarray(Y), np.asfortranarray(D), L=k, numThreads=numThreads)
    W = np.asarray(W.todense())
    return W

'''
Sparse decoding 2
Lasso algrithm
'''
# # Lasso algorithm
# Input (Ver compressed): 
#     Y: Compressed measurement (np.array: Compressed channels x Cells)
#     D: Compressed dictionary of gene modules (np.array: Compressed channels x Modules)
#     lda: Error tolerance coefficient. Per column, it gives sparsest w within error(||y-Dw||_2^2 <= lda*(Y_l2norm)^2/Ncol)
# Input (Ver not-compressed): 
#     Y: Original measurement (Proteins x Cells)
#     D: Dictionary of gene modules (Proteins x Modules)
#     lda: Error tolerance coefficient. Per column, it gives sparsest w within error(||y-Dw||_2^2 <= lda*(Y_l2norm)^2/Ncol)
# Output: 
#     W: Module expression (Modules x Cells)


# return sparsest W such that Y = DW, within column-wise error tolerance upto lda*(Y_l2norm)^2/Ncol.

def sparse_decode_lasso(Y, D, lda, numThreads, mode=1, nonneg=True):
    W = spams.lasso(np.asfortranarray(Y), np.asfortranarray(D), lambda1=lda*(np.linalg.norm(Y)**2/Y.shape[1]) ,
              mode=mode, numThreads=numThreads, pos=nonneg)
    W = np.asarray(W.todense())
    return W  

# # Lasso decoding in blocks
# Group columns into blocks of num_blocks, according to thier l2_norm. (Grouped by simillar l2_norm columns)
# Additional input:
#     num_blocks: number of blocks

def sparse_decode_blocks_lasso(Y, D, lda, numThreads, num_blocks, mode=1,  nonneg=True):
    W = np.zeros((D.shape[1], Y.shape[1]))
    Yl2 = np.linalg.norm(Y, axis=0)
    xs = np.argsort(Yl2)
    block_size = int(len(xs) / num_blocks)
    if (block_size!=0):
        for i in range(0, len(xs), block_size):
                idx = xs[i:i+block_size]   # grouped by the l2_norm of the column
                w = sparse_decode_lasso(Y[:, idx], D, lda, numThreads, mode, nonneg)
                W[:, idx] = w
    else:
            W = sparse_decode_lasso(Y, D, lda, numThreads, mode, nonneg)
    return W
    

'''
Iterative decoding with re-weighing A 
'''
def iterative_decode_W_A_lasso(Y, A, U, ldaW, ldaA, maxItr, nTh, nBl, eachItr = False, mode=1,  nonneg=True):
    Ws = []
    Ahats = [] # used when saving each Itr
    Ahat = A*1 # initialise Ahat as A_input
    for _ in range(maxItr):        
        if eachItr:
            Ahats.append(Ahat)
        W = sparse_decode_blocks_lasso(Y = Y, D = Ahat.dot(U), lda=ldaW, numThreads = nTh, 
                                       num_blocks = nBl, mode=1,nonneg=True)
        Ahat = sparse_decode_lasso(Y = Y.T, D= U.dot(W).T, lda=ldaA, numThreads = nTh, mode=1, nonneg=True).T
        Ahat[A == 0] = 0
        if eachItr:
            Ws.append(W)
    if eachItr:
        return Ws, Ahats
    return W, Ahat



'''
Weighted Lasso (not working yet)
'''
def sparse_decode_weighted_lasso(Y, D, Weight, lda, numThreads, mode=1, nonneg=True):
    Ynorm = np.linalg.norm(Y)**2 / Y.shape[1]
    W = spams.lassoWeighted(np.asfortranarray(Y), np.asfortranarray(D), np.asfortranarray(Weight), lambda1=lda*Ynorm,
              mode=mode, numThreads=numThreads, pos=nonneg)
    W = np.asarray(W.todense())
    # fit = 1 - np.linalg.norm(Y - D.dot(W))**2 / np.linalg.norm(Y)**2
    return W  

def sparse_decode_blocks_weighted_lasso(Y, D, Weight, lda, mode=1, numThreads=20, nonneg=True, num_blocks=20, doprint=True):
        W = np.zeros((D.shape[1], Y.shape[1]))
        Yl2 = np.linalg.norm(Y, axis=0)
        xs = np.argsort(Yl2)
        block_size = int(len(xs) / num_blocks)
        if (block_size!=0):
            for i in range(0, len(xs), block_size):
                    idx = xs[i:i+block_size]
                    w = sparse_decode_weighted_lasso(Y[:, idx], D, Weight[:,idx], lda, numThreads, mode, nonneg, doprint)
                    W[:, idx] = w
        else:
                W = sparse_decode_weighted_lasso(Y, D, lda, numThreads, mode, nonneg, doprint)
        return W
    
'''
random A generator

'''
    
import time
from scipy.spatial import distance    

# Produce random A
# see below "produce_random_As" for details of the inputs

def random_phi_subsets_g(m, g, n, d_thresh=0.4, pmax = 0.9):
    Phi = np.zeros((m, g))
    Phi[np.random.choice(m,np.random.randint(n[0], n[1]+1), replace=False), 0] = 1
    for i in range(1, g):
        dmax = 1
        # Set starting time to make sure while loop is not stuck in infinit loop
        start = time.time()
        # Add random columns with at most n[1] 1's which have a min distance from
        # last column of 1 - dist > d_thresh (such that columns are not to similar)
        # until Phi has g columns
        while dmax > d_thresh:
            p = np.zeros(m)
            # create probability list for possible ncomposite per gene 
            # prob should look like this: [0.05,0.05,0.9] when maxcomposition is 3
            prob = [0]*n[1]
            prob[-1] = pmax
            prob[:-1] = [(1-pmax)/(n[1]-1)]*int(n[1]-1)
            # create a random column for phi
            p[np.random.choice(m,np.random.choice(n[1],p=prob)+1, replace=False)] = 1
            # calc. dmax
            dmax = 1 - distance.cdist(Phi[:,:i].T, [p], 'correlation').min()
            # print('dmax:' ,dmax)
            # Catch that while loop is not stuck in infinite loop
            end = time.time()
            if (end-start) > 60:
                raise ValueError(('Computing Phi/A took to long or got stuck ' +
                                  'in an infinit loop. Either there are not enough ' +
                                  'combinations possible using nmeasurements={0}'.format(m) +
                                  ', # proteins={0} and '.format(g) +
                                  'maxcomposition={0} or d_thresh is to small.'.format(n[1])))
        Phi[:,i] = p
    # Scale between 0-1 per column (protein) as is more realistic in an experiment
    Phi = Phi / Phi.sum(0)
    return Phi


# Check that no channel is unused or that difference between the number of times
# each protein is used is not too big
def check_balance(Phi, thresh=4):
    x = Phi.sum(0) + 1e-7
    if (x.max() / x.min() > thresh) or (Phi.sum(1).min() == 0):
        return False
    else:
        return True    
    

# # Produce random As 
# In total, [n_A_each*n_L0sum] As are produced.
# n_A_each: n of random As produced per unique L0sum
# n_L0sum: n of unique L0 sum. Defined number of L0sum will be selected in desending order from the max L0sum.
# maxp (default = 1): initial prob of choosing n_max
# g: no of total genes
# m: n of composite channels 
# n: [min,max] of channels per gene
# d_thresh: distance threshold per gene
# maxp_inc (default = 0.05): incriment of probability adjustment per round.

def produce_random_As(n_A_each, n_L0sum, m, n, d_thresh, g, maxp = 1, maxp_inc = 0.05):
    # set params for random As
    n_A_each = n_A_each  # n of random As produced
    n_L0sum = n_L0sum    # n of unique L0 sum (from the max)

    m = m                # n of composite channels 
    n = n                # [min,max] of channels per gene
    d_thresh = d_thresh  # distance threshold per gene
    g = g                # no of total genes
    maxp = maxp          # initial prob of choosing n_max
    maxp_inc = maxp_inc    # incriment of probability adjustment per round.

    L0sum_max = n[1]*g
    L0sum_min = n[1]*g-n_L0sum+1
    print("Random {} As with L0sum between {} and {}. {} genes into {} channels, max {} channels per gene".format(
        n_A_each*n_L0sum, L0sum_min,L0sum_max, g, m, n[1]))
    
    # produce random As
    counter = [0]*n_L0sum
    Phi = []
    L0Sums = []
    overflow_count_within = 0
    overflow_count_below = 0
    
    while min(counter) < n_A_each:
        # create phi
        while True:
            phi = random_phi_subsets_g(m, g, n, d_thresh,maxp)
            if check_balance(phi):
                break
        L0sum = np.linalg.norm(phi,ord = 0, axis = 1).sum() 
        idx = int(L0sum_max-L0sum)  # index for counter_list (max L0sum at index 0)
        # if L0 sum was larger than defined minimum
        if L0sum >= L0sum_min:    
            # if already enough number of phi was created for this L0sum, modify the pmax and try again. 
            if counter[idx] >= n_A_each: 
                overflow_count_within += 1
                if idx <= (n_L0sum-1)/2: # if overflowed L0sum was larger than average reduce pmax
                    maxp -= maxp_inc
                elif maxp+maxp_inc <= 1: # if overflowed L0sum was smaller than average increase pmax
                    maxp += maxp_inc
                continue
            # Add to Phi and counter plus 1
            else:
                counter[idx] += 1
                Phi.append(phi)
                L0Sums.append(L0sum)
        # if L0 sum was smaller than defined minimum increase pmax (probability for choosing n_max)
        elif maxp+maxp_inc <= 1:
            maxp += maxp_inc
            overflow_count_below += 1

    print(overflow_count_within,overflow_count_below)
    return Phi
    
'''
Simulate Y from A.X

'''
def get_observations(X0, Phi, inverted_snr=1/5, return_noise=False, normalization='none'):
    # Extract data matrix X from anndata object and apply selected normalization
    match normalization:
        case 'paper_norm':
            # Normalize X? (otherwise spams.nmf underneath will estimate U to only contain
            # 0s and smaf won't work)
            # Normalizaiton: Every row in X_input is divided by the corresponding vector
            # element in the rowwise norm (proteins are normalized accross all cells/pixels)
            X = (X0.T / np.linalg.norm(X0, axis=1)).T
        case 'min_max_norm':
            X = (X0-X0.min(axis=1, keepdims=True)) / (
                X0.max(axis=1, keepdims=True)-X0.min(axis=1, keepdims=True)
                )
        # case 'zscore_norm':
        #     X = ((X_mat.T-np.mean(X_mat, axis=1)) / np.std(X_mat, axis=1)).T # zscore(X_mat, axis=1)
        case 'none':
            X = X0
        case _:
            # In case no valid normalization is given, an error is thrown
            raise ValueError(('The normalization {0} used by smaf is not valid.'.format(normalization) +
                              'Please use one of the following: paper_norm, min_max_norm, ' +
                              'or none.'))

    noise = np.array([np.random.randn(X.shape[1]) for _ in range(X.shape[0])])
    noise *= np.linalg.norm(X)/np.linalg.norm(noise)*inverted_snr
    if return_noise:
        return Phi.dot(X + noise), noise
    else:
        return Phi.dot(X + noise)

'''
Compare Xtrue and Xsim
'''
from scipy.stats import spearmanr, pearsonr

def compare_results(A, B):
    results = list(correlations(A, B, 0))[:-2]
    results += list(compare_distances(A, B))
    results += list(compare_distances(A.T, B.T))
    return results
colnames = ['version', 'Overall pearson', 'Overall spearman', 'Gene average',
            'Sample average', 'Sample dist pearson', 'Sample dist spearman',
            'Gene dist pearson', 'Gene dist spearman',
            'Matrix coherence (90th ptile)']
# Compare distance between random columns of two matrices
def compare_distances(A, B, random_samples=[], s=200, pvalues=False):
	if len(random_samples) == 0:
		random_samples = np.zeros(A.shape[1], dtype=np.bool)
		random_samples[:min(s, A.shape[1])] = True
		np.random.shuffle(random_samples)
	dist_x = distance.pdist(A[:, random_samples].T, 'euclidean')
	dist_y = distance.pdist(B[:, random_samples].T, 'euclidean')
	pear = pearsonr(dist_x, dist_y)
	spear = spearmanr(dist_x, dist_y)
	if pvalues:
		return pear, spear
	else:
		return pear[0], spear[0]

# Compute correlations between two matrices: between cells (/pixels),
# between proteins, overall and SVD of matrices
def correlations(A, B):
# 	overall_p = (1 - distance.correlation(A.flatten(), B.flatten()))  #overall pearson
# 	overall_s = spearmanr(A.flatten(), B.flatten())               #overall spearman
	genewise_p = np.zeros(A.shape[0])
	for i in range(A.shape[0]):
		genewise_p[i] = 1 - distance.correlation(A[i], B[i])  #gene-wise pearson
	geneaverage_p = np.average(genewise_p[np.isfinite(genewise_p)])    #gene average pearson
	genemin_p = np.nanmin(genewise_p[np.isfinite(genewise_p)],where=genewise_p[np.isfinite(genewise_p)].size>0, initial=np.nan)  #gene minimum pearson
	Anorm = (A.T/np.linalg.norm(A, axis=1)).T
	Bnorm = (B.T/np.linalg.norm(B, axis=1)).T
	cellwise_p = np.zeros(Anorm.shape[1])
	for i in range(Anorm.shape[1]):
		cellwise_p[i] = 1 - distance.correlation(Anorm[:,i], Bnorm[:,i]) #cell-wise pearson
	cellaverage_p = np.average(cellwise_p[np.isfinite(cellwise_p)])  #cell average pearson
# 	pc_dist = [] # default pc_n was 100
# 	if pc_n > 0:
# 		u0, s0, vt0 = np.linalg.svd(A)
# 		u, s, vt = np.linalg.svd(B)
# 		for i in range(pc_n):
# 			pc_dist.append(abs(1 - distance.cosine(u0[:,i], u[:,i])))
# 		pc_dist = np.array(pc_dist)
	return [geneaverage_p, genemin_p, cellaverage_p], [genewise_p, cellwise_p] #pc_dist

'''
Analyse properties of U, W, and how well the fit (UW~X) was.
'''
def analyse_U_W(U, W, Xtrue):
    # U properties
    U_l1 = np.linalg.norm(U, ord = 1, axis = 0).mean() #mean of module-wise l1 norm 
    U_l0 = np.linalg.norm(U, ord = 0, axis = 0).mean() #mean of module-wise l0 norm (how many genes are non-zero per module)
    d_num = U.shape[1]                                 #no of non-zero modules in the end
    C_90 = np.percentile(1-distance.pdist(U.T,'cosine'),90) # 90th percentile coherence
    # W properties
    W_l0 = np.linalg.norm(W, ord = 0, axis = 0).mean() #mean of cell-wise l0 norm
    # calc. fit
    A = U.dot(W)
    B = Xtrue
    Fit = 1 - (np.linalg.norm(B-A)**2/ np.linalg.norm(B)**2)
    # summerise results
    results = [U_l1,U_l0,d_num, C_90, W_l0, Fit]
    colnames = ['U_l1_mean', 'U_l0_mean','d_modules','U_90p_coherence','SMAF_W_l0_mean','SMAF_Fit']
    return results, colnames
    
    
# For correlations of X and UW after SMAF.
def analyse_U_W_corr(U, W, Xtrue, detail=False):    
    # correlations
    A = U.dot(W)
    B = Xtrue    
    results,details = correlations(A,B)
    colnames = ['SMAF_Gene_average','SMAF_Gene_minimum','SMAF_Cell_average']
    # Return gene_wise/cell_wise corr. if requested.
    if detail:
        return results, colnames, details # details are genewise and cellwise pearson     
    return results, colnames

'''
For decompression of W from Y = AUW.
Analyse correlations of X and Xhat (and W and AU), and how well the fit (Xhat~X) was.

name: suffix for the returned "colnames".
detail: If True, a list of genewise correlation and cellwise correlation is also returned (as "details")
'''
def analyse_decoding(phi,U,W,Xtrue, Xhat, name="", detail=False):
    # correlations
    A = Xhat
    B = Xtrue    
    results,details = correlations(A,B)
    Fit = 1 - (np.linalg.norm(B-A)**2/ np.linalg.norm(B)**2)
    W_l0 = np.linalg.norm(W, ord = 0, axis = 0).mean() #mean of cell-wise l0 norm
    C_90 = np.percentile(1-distance.pdist(phi.dot(U).T,'cosine'),90) # 90th percentile coherence of AU
    results.extend([Fit,W_l0,C_90])
    colnames = [name+'Gene_average',name+'Gene_minimum',name+'Cell_average',name+'Fit',name+'W_l0_mean','AU_90p_coherence']
    # Return gene_wise/cell_wise corr. if requested.
    if detail:
        return results, colnames, details # details are genewise and cellwise pearson     
    return results, colnames


'''
scatter plotting Xtrue vs Xhat
'''
import matplotlib.pyplot as plt

def plot_Xtrue_Xhat(Xtrue,Xhat,gene_order,gene_corr):
    ntotal = Xhat.shape[0]
    nrow = int(np.ceil(ntotal**0.5))
    ncol = int(np.floor(ntotal**0.5))
    fig, axs = plt.subplots(nrow, ncol, tight_layout=True, figsize = (10,10))
    for k in range(ntotal):
        i = k//nrow
        j = k%nrow
        axs[i,j].scatter(Xtrue[k,:],Xhat[k,:],s=0.01 * fig.get_figheight())
        axs[i,j].set_title("{}_corr:{:.3f} ".format(gene_order[k],gene_corr[k]))
    fig.supxlabel('ground truth')
    fig.supylabel('Decompressed')





