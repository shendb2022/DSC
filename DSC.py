import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import time


def standard(X):
    '''
    Standardization
    universal
    :param X:
    :return:
    '''
    min_x = np.min(X)
    max_x = np.max(X)
    if min_x == max_x:
        return np.zeros_like(X)
    return np.float32((X - min_x) / (max_x - min_x))


def soft(X, lamd):
    '''
    Soft thresholding function
    :param X: input
    :param lamd: threshold
    :return:
    '''
    t1 = np.sign(X)
    t2 = np.abs(X) - lamd
    t2[t2 < 0] = 0
    return t1 * t2


def SAMDetector(X, P):
    '''
    SAM for target detection, the smaller the angle is, the more similar the two vectors are
    :param X: The image X in matrix form L*N
    :param P: The target prior P with a size of L*1
    :return:
    '''
    norm_X = np.sqrt(np.sum(np.square(X), axis=0))
    norm_P = np.sqrt(np.sum(np.square(P), axis=0))

    x_dot_P = np.sum(X * P, axis=0)

    value = x_dot_P / (norm_X * norm_P)

    angle = np.arccos(np.clip(value, -1, 1))
    return angle * 180 / np.pi


def readData(data_no):
    '''
    read data
    :param data_no: 0 means San Diego, 1 means HYDICE
    :return:
    '''
    if data_no == 0:
        path = 'Sandiego.mat'
        target_prior_coordinates = (22, 69)
        Kt = 20
    elif data_no == 1:
        path = 'HYDICE.mat'
        target_prior_coordinates = (65, 36)
        Kt = 130
    else:
        print('invalid data')
        return

    mat = sio.loadmat(path)
    hs = mat['data']
    groundtruth = mat['map']

    if data_no == 0:
        # The water-absorption bands in San Diego should be removed firstly
        # while in HYDICE, the low-SNR bands have been already removed
        hs = np.concatenate(
            [hs[:, :, 6:32], hs[:, :, 35:96], hs[:, :, 97:106], hs[:, :, 113:152], hs[:, :, 166:220], hs[:, :, 224:]],
            axis=-1)
    ## standardized the pixel values to the range of [0,1]
    hs = standard(hs)
    H, W, L = hs.shape

    ## matrix form
    hs_matrix = np.reshape(hs, [-1, L], order='F')
    hs_matrix = hs_matrix.T  ## L * N

    ## select the target prior
    target_prior = hs[target_prior_coordinates[0], target_prior_coordinates[1]]
    target_prior = np.expand_dims(target_prior, axis=-1)  ## L * 1

    return hs_matrix, groundtruth, target_prior, H, W, L, Kt


def dictionaryConstruction(X, p, Kt, m=10, n=25):
    '''
    construction of target dictionary and background dictionary
    :param X: The original hyperspectral image with size of L*N
    :param p: The target prior with size of L*1
    :param Kt: The number of atoms in the target dictionary
    :param m: The number of classes of background pixels using K-means
    :param n: The number of picked background atoms in each class
    :return:.
    '''
    ## construction of the target dictionary At
    L, N = X.shape
    # calculate the SAM values of all pixels
    detection_result = SAMDetector(X, p)
    # sort the scores in ascending order
    ind = np.argsort(detection_result)
    ind = ind.flatten(order='F')

    # select the first Kt pixels as the atoms of At
    ind = ind[:Kt]
    At = X.T[ind]

    # update X
    target_map = np.zeros([N])
    target_map[ind] = 1
    X = X.T[target_map == 0]

    ## construction of the background dictionary Ab

    # cluster pixels in X into m classes using k-means
    estimator = KMeans(n_clusters=m)
    estimator.fit(X)
    idx = estimator.labels_

    # calculate the number of samples in each class
    N = np.zeros(shape=[m], dtype=np.int32)
    for i in range(m):
        N[i] = len(np.where(idx == i)[0])

    # calculate the mean of samples in each class
    Xmeans = np.zeros(shape=[m, L], dtype=np.float32)
    for i in range(m):
        Xmeans[i, :] = np.mean(X[idx == i], axis=0)

    Ab = []
    R = []
    for i in range(m):
        if N[i] < L:  #
            continue
        cind = np.where(idx == i)[0]
        Xi = X[cind]
        rXi = Xi - Xmeans[i, :]

        # calculate the covariance of samples in each class
        cov = np.matmul(rXi.T, rXi) / (N[i] - 1)
        incov = np.linalg.inv(cov)

        # calculate the Mahalanobis distance of each sample in each class
        for j in range(N[i]):
            mdj = rXi[j, :].dot(incov).dot(rXi[j, :].T)
            R.append(mdj)

        # sort R in ascending order
        ind = np.argsort(R)

        # select the first n pixels in each class
        Ab.append(X[cind[ind[:n]]])
        R.clear()

    Ab = np.concatenate(Ab, axis=0)
    return Ab.T, At.T


def modelSolver(X, Ab, At, lambd):
    '''
    Estimate the target coefficients St with size of Kt*N
    :param X: The original hyperspectral image with size of L*N
    :param Ab: The background dictionary with size of L*Kb
    :param At: The target dictionary with size of L*Kt
    :param lambd: the trade-off parameter
    :return: St
    '''

    maxIter = 1e6
    L, N = X.shape
    Kb = Ab.shape[-1]
    Kt = At.shape[-1]
    gama = 1.1
    max_mu = 1e30
    mu = 1
    epsilon = 1e-5

    # To be optimized variables
    Sb = np.zeros([Kb, N], dtype=np.float32)
    St = np.zeros([Kt, N], dtype=np.float32)

    # Three Lagrangian multipliers
    H1 = np.zeros_like(X, dtype=np.float32)
    H2 = np.zeros_like(Sb, dtype=np.float32)
    H3 = np.zeros_like(St, dtype=np.float32)

    # To avoid repeated calculations
    abtx = np.matmul(Ab.T, X)
    attx = np.matmul(At.T, X)

    inv_ab = np.linalg.inv(np.matmul(Ab.T, Ab) + np.eye(Kb))
    inv_at = np.linalg.inv(np.matmul(At.T, At) + np.eye(Kt))

    AtSt = np.zeros_like(X)

    ## The error of two adjacent iterations
    stopC = 1
    ## Iterations will be stopped when the error of two adjacent iterations is smaller than the epsilon

    for iter in range(int(maxIter)):
        # update Vb
        Vb = soft(Sb + H2 / mu, 1 / mu)

        # update Sb
        Sb = np.matmul(inv_ab, abtx - Ab.T.dot(AtSt) + Vb + (Ab.T.dot(H1) - H2) / mu)
        AbSb = np.matmul(Ab, Sb)

        # update Vt
        Vt = soft(St + H3 / mu, lambd / mu)

        # update St
        St = np.matmul(inv_at, attx - At.T.dot(AbSb) + Vt + (At.T.dot(H1) - H3) / mu)
        AtSt = np.matmul(At, St)

        leq1 = X - AbSb - AtSt
        leq2 = Sb - Vb
        leq3 = St - Vt

        pre_stopC = stopC

        stopC1 = max(np.max(leq1), np.max(leq2))
        stopC = max(np.max(leq3), stopC1)
        stop_diff = abs(stopC - pre_stopC)

        if iter == 0 or iter % 50 == 0 or stop_diff < epsilon:
            print('iter %s  mu= %s  stopALM= %s ' % (iter, mu, stop_diff))

        if stop_diff < epsilon:
            print('iter %s  mu= %s  stopALM= %s ' % (iter, mu, stop_diff))
            break
        else:
            H1 = H1 + mu * leq1
            H2 = H2 + mu * leq2
            H3 = H3 + mu * leq3
            mu = min(max_mu, mu * gama)
    return St


if __name__ == '__main__':
    start = time.perf_counter()
    ## read data, 0 means San Diego, and 1 means HYDICE
    X, gt, p, H, W, L, Kt = readData(0)
    ## construct the background dictionary and target dictionary
    Ab, At = dictionaryConstruction(X, p, Kt=Kt, m=10, n=25)

    ## estimate the target coefficent by solving the optimizing model
    St = modelSolver(X, Ab, At, lambd=1)
    ## get the target image with target dictionary and coefficents
    T = At.dot(St)
    T = np.clip(T, 0, 1)
    ## get the detection result using L2-norm in the spectral dimension
    detection_map = np.sqrt(np.sum(T ** 2, axis=0))

    end = time.perf_counter()
    ## calculte the running time
    print('time consumes %sseconds' % (end - start))

    y_l = np.reshape(gt, [-1, 1], order='F')
    y_p = detection_map.T

    ## calculate the AUC value
    auc = metrics.roc_auc_score(y_l, y_p)
    ap = metrics.average_precision_score(y_l, y_p)
    print(auc)
    print(ap)

    ## visulize the detection result
    detection_map = np.reshape(detection_map, [H, W], order='F')
    plt.imshow(detection_map,cmap='gray')
    plt.show()
