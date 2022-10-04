import datetime, sqlite3, os, os.path, scipy, math, pickle, sys, random, shlex, subprocess, inspect
from termcolor import cprint
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
import numpy as np
import pickle
from pytz import timezone
import numpy as np
from numpy import savetxt, loadtxt
import traj_dist, traj_dist.distance
import torch

import sklearn, sklearn.mixture
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, SpectralClustering, OPTICS, DBSCAN
# import sklearn_extra
# import sklearn_extra.cluster


def shCommand(text):
    cmd = shlex.split(text)
    subprocess.run(cmd)


def Time(text="time", prnt=True, color='green', on_color='on_grey'):
    tz = timezone('Asia/Shanghai')
    dt = datetime.datetime.now(tz)
    if prnt:
        cprint(text + ": " + str(dt), color=color, on_color=on_color)
    return dt


class DistFuncs():

    def Lcss(self, X, Y, L, param):
        return traj_dist.distance.c_e_lcss(X, Y, param)

    def Dtw(self, X, Y, L, param):
        return traj_dist.distance.c_e_dtw(X, Y)

    def Hausdorf(self, X, Y, L, param):
        return traj_dist.distance.c_e_hausdorff(X, Y)

    # def Frechet(self, X, Y, L, param):
    #     return traj_dist.distance.c_frechet(X,Y)

    # def Sowd_grid(self, X, Y, L, param):
    #     return traj_dist.distance.c_sowd_grid(X,Y)

    # def Erp(self, X, Y, L, param):
    #     return traj_dist.distance.c_e_erp(X,Y, )

    def Edr(self, X, Y, L, param):
        return traj_dist.distance.c_e_edr(X, Y, param)

    def Sspd(self, X, Y, L, param):
        return traj_dist.distance.c_e_sspd(X, Y)


def DistMatricesSection(trajectories=None, nTraj=None, dataName="mot15", pickleInDistMatrix=False, test=True,
                        maxTrajLength=1800,
                        similarityMeasure=None,
                        lcssParamList=None, pfParamList=None):
    if pfParamList is None:
        pfParamList = [0.1, 0.2, 0.3, 0.4]
    if lcssParamList is None:
        lcssParamList = [1, 2, 3, 5, 7, 10]
    if similarityMeasure is None:
        similarityMeasure = [['Lcss', [1, 2, 3, 5, 7, 10]], ['Dtw', [-1]], ['Hausdorf', [-1]],
                             ['Edr', [1, 2, 3, 5, 7, 10]], ['Sspd', [-1]],
                             ]
        # similarityMeasure = [['GuLcss', [1, 2, 3, 5, 7, 10]], ['GuDtw', [-1]],
        #                      ['GuPf', [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]],
        #                      ['Lcss', [1, 2, 3, 5, 7, 10]], ['Dtw', [-1]], ['Hausdorf', [-1]],
        #                      ['Edr', [1, 2, 3, 5, 7, 10]], ['Sspd', [-1]],
        #                      ]
    file_path = "./output"
    try:
        os.mkdir(f"{file_path}/data")
    except:
        pass
    # !mkdir ./data

    if nTraj is None:
        nTraj = len(trajectories)

    if pickleInDistMatrix:
        shCommand("wget -O {}/data/distMatricesZip.zip".format(file_path))
        shCommand("rm -r -d {}/data/distMatricesFolder".format(file_path))
        shCommand("unzip -o {}/data/distMatricesZip.zip".format(file_path))

    else:
        shCommand("rm -d -r {}/data/distMatricesFolder".format(file_path))
        shCommand("mkdir {}/data/distMatricesFolder".format(file_path))
        distMatrices = []
        LZero = np.zeros((maxTrajLength + 1, maxTrajLength + 1))  ######################
        LInf = np.full((maxTrajLength + 1, maxTrajLength + 1), 1e6)
        distFuncs = DistFuncs()
        # testTraj0 = np.array([[0,0],[0,1]], dtype=float)
        # testTraj1 = np.array([[0.75,0],[0.5,1],[1,1.5],[0.75,2.5]], dtype=float)

        for (methodName, method) in inspect.getmembers(distFuncs):
            # try:
            #     method(testTraj0, testTraj1, L, 0.5)+0
            for [distName, paramValueList] in similarityMeasure:
                if methodName == distName:
                    if distName in ['Dtw', 'GuDtw', 'GuPf']:
                        LMatrix = LInf.copy()
                    else:
                        LMatrix = LZero.copy()
                    for paramValue in paramValueList:
                        distMatrix = np.zeros((nTraj, nTraj))
                        startTime = Time(prnt=False)
                        for i in range(nTraj):
                            for j in range(i + 1):
                                tr1 = trajectories[i]
                                tr2 = trajectories[j]
                                distMatrix[i, j] = method(tr1, tr2, LMatrix, paramValue)
                                distMatrix[j, i] = distMatrix[i, j]

                        endTime = Time(prnt=False)
                        print(f'{distName}, paramValue={paramValue}, runtime:{str(endTime - startTime)}')
            
                        savetxt(f"{file_path}/data/distMatricesFolder/{dataName}_{distName}Matrix_param{paramValue}.csv",
                                distMatrix, delimiter=',')
                        distMatrices.append((distMatrix, f"{dataName}_{distName}Matrix_param{paramValue}"))
    
    if not test:
        pickle_out = open(f'{file_path}/mot15/distances/distMatrices.pickle', "wb")
        pickle.dump(distMatrices, pickle_out)
        pickle_out.close()

    return distMatrices


def OdClustering(funcTrajectories=None, nTraj=None, dataName="mot15", modelList=None, nClusDestSet=[4],
                 modelNames=['average Agglo-Hierarch'], nIter=1, visualize=False, shuffle=False, test=True,
                 darkTheme=False):
    if shuffle or len(nClusDestSet) > 1:
        cprint(
            '\n The internal set of trejecories and thus the output labels go out of sync with the input "trajectories" set if shuffle==True or len(nClusOriginSet)>1 or len(nClusDestSet)>1.\n',
            color='red', on_color='on_grey')

    if darkTheme:
        tickColors = 'white'
    else:
        tickColors = 'black'

    if nTraj == None:
        nTraj = len(funcTrajectories)

    startPoints = np.zeros((nTraj, 2))
    endPoints = np.zeros((nTraj, 2))
    for i in range(nTraj):
        tr = funcTrajectories[i]
        startPoints[i] = tr[0]
        endPoints[i] = tr[-1]

    # startAvgDists = np.zeros((len(nClusOriginSet), nIter))
    endAvgDists = np.zeros((len(nClusDestSet), nIter))
    minTrajX = min([min(tr[:, 0]) for tr in funcTrajectories])
    minTrajY = min([min(tr[:, 1]) for tr in funcTrajectories])
    maxTrajX = max([max(tr[:, 0]) for tr in funcTrajectories])
    maxTrajY = max([max(tr[:, 1]) for tr in funcTrajectories])

    if modelList == None:
        modelList = [
            # (sklearn_extra.cluster.KMedoids(metric='euclidean'), 'KMedoids'),
            (KMeans(), 'KMeans')
            , (AgglomerativeClustering(affinity='euclidean', linkage='ward'), 'ward Agglo-Hierarch')
            , (AgglomerativeClustering(affinity='euclidean', linkage='complete'),
               'complete Agglo-Hierarch')
            ,
            (AgglomerativeClustering(affinity='euclidean', linkage='average'), 'average Agglo-Hierarch')
            , (AgglomerativeClustering(affinity='euclidean', linkage='single'), 'single Agglo-Hierarch')
            , (Birch(threshold=0.5, branching_factor=50), 'BIRCH')
            , (
                sklearn.mixture.GaussianMixture(covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=200,
                                                n_init=5,
                                                init_params='kmeans'), 'GMM')
            , (SpectralClustering(affinity='rbf', n_jobs=-1), 'Spectral')
            , (OPTICS(metric='minkowski', n_jobs=-1), 'OPTICS')
            , (DBSCAN(metric='euclidean', n_jobs=-1), 'DBSCAN')
        ]

    pickedModels = [(model, title) for (model, title) in modelList if title in modelNames]

    for model, title in pickedModels:
        for iter in range(nIter):
            shufIndices = [i for i in range(len(startPoints))]
            if shuffle or len(nClusDestSet) > 1:
                random.shuffle(shufIndices)
            shufStartPoints = np.zeros_like(startPoints)
            shufEndPoints = np.zeros_like(endPoints)
            for i in range(len(shufIndices)):
                shufStartPoints[shufIndices[i]] = startPoints[i]
                shufEndPoints[shufIndices[i]] = endPoints[i]

            shufStartPoints = startPoints.copy()
            shufEndPoints = endPoints.copy()
            if shuffle or len(nClusDestSet) > 1:
                # random.shuffle(shufStartPoints)
                random.shuffle(shufEndPoints)

            # t = Time(text=title)
            for i in range(len(nClusDestSet)):
                # t = Time("nClus={}".format(nClusOriginSet[i]))
                model2 = model
                try:
                    if title == "GMM":
                        model2.n_components = nClusDestSet[i]
                    else:
                        model2.n_clusters = nClusDestSet[i]
                except:
                    pass
                endModel = model2.fit(shufEndPoints)
                try:
                    endLabels = list(endModel.labels_)
                except:
                    endLabels = list(endModel.predict(shufEndPoints))
                endLabelList = list(set(endLabels))
                nClusEnd = len(endLabelList)
                try:
                    endCenters = endModel.cluster_centers_
                except:
                    endCenters = np.zeros((nClusEnd, 2))
                    for k in range(nClusEnd):
                        clusPoints = shufEndPoints[endLabels == endLabelList[k]]
                        endCenters[k] = np.average(clusPoints, axis=0)
                endDistSum = 0
                for k in range(nClusEnd):
                    clusPoints = shufEndPoints[endLabels == endLabelList[k]]
                    for point in clusPoints:
                        endDistSum += ((point[0] - endCenters[k, 0]) ** 2 + (point[1] - endCenters[k, 1]) ** 2) ** 0.5
                endAvgDists[i, iter] = endDistSum / len(endLabels)
                # print(endAvgDists[i])
                if visualize:  # and len(nClusOriginSet)>1 and len(nClusDestSet)>1:
                    try:
                        print("Calinski Harabasz: {}".format(
                            sklearn.metrics.calinski_harabasz_score(shufEndPoints, endLabels)))
                    except:
                        pass
                    print("nClusEnd={}".format(nClusEnd))
                    cmap = list(colors.TABLEAU_COLORS)
                    colormap = cmap
                    repeat = nClusEnd // len(cmap)
                    for k in range(repeat):
                        colormap = colormap + cmap
                    plt.figure(figsize=(8, 6))
                    for k in range(nTraj):
                        plt.scatter(shufEndPoints[k, 0], shufEndPoints[k, 1], c=colormap[endLabels[k]])
                    plt.scatter(endCenters[:, 0], endCenters[:, 1], c='black')
                    plt.tick_params(colors=tickColors)

                    plt.savefig("./../output/{}.jpg".format(title))
                    # plt.show()
    #
    # meanStartAvgDist = np.mean(startAvgDists, axis=-1)
    # stdStartAvgDist = np.std(startAvgDists, axis=-1)  # /(nIter**0.5)
    meanEndAvgDist = np.mean(endAvgDists, axis=-1)
    stdEndAvgDist = np.std(endAvgDists, axis=-1)  # /(nIter**0.5)
    plotMax = np.max(meanEndAvgDist + stdEndAvgDist)

    if len(nClusDestSet) > 1:
        plt.figure(figsize=(16, 12))

        fig = plt.subplot(2, 1, 1)
        fig.set_ylim(0, plotMax)
        fig.errorbar(x=nClusDestSet, y=np.mean(endAvgDists, axis=-1),
                     yerr=np.std(endAvgDists, axis=-1) / (nIter ** 0.5), linestyle='None', fmt='-_', color='orange',
                     ecolor='blue')
        fig.tick_params(colors=tickColors)
        fig.set_title("Destination clusters")
        plt.show()

    if not test:
        try:
            os.mkdir("./data/" + dataName + "_output")
        except:
            pass
        # savetxt('./data/' + dataName + "_startLabels.CSV", startLabels, delimiter=',')
        savetxt('./data/' + dataName + "_endLabels.CSV", endLabels, delimiter=',')

    # return startLabels, endLabels, shufStartPoints, shufEndPoints
    # unShufStartLabels = np.zeros_like(startLabels)
    unShufEndLabels = np.zeros_like(endLabels)
    for i in range(len(shufIndices)):
        # unShufStartLabels[i] = startLabels[shufIndices[i]]
        unShufEndLabels[i] = endLabels[shufIndices[i]]

    return unShufEndLabels, endPoints, nClusEnd  # , startAvgDists


def OdMajorClusters(trajectories=None, startLabels=None, endLabels=None, dataName="mot15", threshold=10,
                    visualize=False, test=True,
                    load=False):
    if load:
        try:
            # startLabels = loadtxt('./data/' + dataName + "_startLabels.CSV", delimiter=',')
            endLabels = loadtxt('./data/' + dataName + "_endLabels.CSV", delimiter=',')
        except:
            raise Exception(
                "No such file or directory: ./data/" + dataName + "_endLabels.CSV, " + "./data/" + dataName + "_endLabels.CSV")
    # else:
    #     startLabels, endLabels = startLabels, endLabels

    countOD = np.zeros((len(set(startLabels)), len(set(endLabels))))
    sampleTraj = np.zeros((len(set(startLabels)), len(set(endLabels))))

    startClusterIndices = []
    for i in list(set(startLabels)):
        startClusterIndices.append(list(np.where(startLabels == i)[0]))

    endClusterIndices = []
    for i in list(set(endLabels)):
        endClusterIndices.append(list(np.where(endLabels == i)[0]))

    odTrajLabels = np.full(len(startLabels), -1)
    refTrajIndices = []
    for i in range(countOD.shape[0]):
        for j in range(countOD.shape[1]):
            lst = list(set(startClusterIndices[i]) & set(endClusterIndices[j]))
            countOD[i, j] = len(lst)
            odTrajLabels[lst] = int(i * countOD.shape[1] + j)
            if countOD[i, j] > 0:
                sampleTraj[i, j] = lst[0]
            if countOD[i, j] > threshold:
                refTrajIndices.extend(lst)
    refTrajIndices.sort()

    # refDistMatrix = np.zeros((len(refTrajIndices), len(refTrajIndices)))
    # for i in range(len(refTrajIndices)):
    #     refDistMatrix[i] = distMatrix[refTrajIndices[i], refTrajIndices]

    if visualize:
        ### major OD visulaization

        # print(countOD)
        threshold = 10
        plt.figure(figsize=(16, 6))
        fig = plt.subplot(1, 2, 1)
        fig.set_title('examples of major ODs', color='w')
        for i in range(countOD.shape[0]):
            for j in range(countOD.shape[1]):
                if countOD[i, j] > threshold:
                    k = int(sampleTraj[i, j])
                    tr = trajectories[k]
                    fig.plot(tr[:, 0], tr[:, 1], label=len(tr))
                    fig.scatter(tr[0, 0], tr[0, 1], c=100, s=2, marker='o')
        fig.legend()
        fig = plt.subplot(1, 2, 2)
        fig.set_title('examples of minor ODs', color='w')
        for i in range(countOD.shape[0]):
            for j in range(countOD.shape[1]):
                if countOD[i, j] <= threshold:
                    k = int(sampleTraj[i, j])
                    tr = trajectories[k]
                    fig.plot(tr[:, 0], tr[:, 1])
                    fig.scatter(tr[0, 0], tr[0, 1], c=100, s=2, marker='o')
        plt.show()

    # if not test:
    #     try:
    #         os.mkdir("./data/" + dataName + "_output")
    #     except:
    #         pass
    #     savetxt('./data/' + dataName + "_output/" + dataName + "_refTrajIndices.CSV", refTrajIndices, delimiter=',')
    #     savetxt('./data/' + dataName + "_output/" + dataName + "_odTrajLabels.CSV", odTrajLabels, delimiter=',')

    return refTrajIndices, odTrajLabels


def Silhouette(model, distMatrix, trajIndices=None):  # , permuting=True):
    if trajIndices == None:
        trajIndices = list(range(distMatrix.shape[0]))
    subDistMatrix = distMatrix[trajIndices][:, trajIndices]

    shufIndices = [i for i in range(len(trajIndices))]
    random.shuffle(shufIndices)
    shufSubDistMatrix = np.zeros_like(subDistMatrix)
    for i in range(len(shufIndices)):
        for j in range(len(shufIndices)):
            # shufSubDistMatrix[i,j] = subDistMatrix[shufIndices[i], shufIndices[j]]
            shufSubDistMatrix[shufIndices[i], shufIndices[j]] = subDistMatrix[i, j]

    model = model.fit(shufSubDistMatrix)
    # labels = model.labels_#[trajIndices]
    shufLabels = model.labels_  # [trajIndices]

    labels = np.zeros_like(shufLabels)
    # print(shufIndices)
    # print(i, np.where(shufIndices==i))
    for i in range(len(shufIndices)):
        labels[np.where(np.array(shufIndices) == i)[0][0]] = shufLabels[i]

    clusters = list(set(labels))
    nClus = len(clusters)
    A = np.zeros(len(labels))
    B = np.full(len(labels), np.inf)
    S = np.zeros(len(labels))
    argmins = np.zeros(len(labels))
    # argmins = [0 for i in range(len(labels))]
    closestCluster = labels
    for i in range(len(labels)):
        similarTrajs = [l for l in range(len(labels)) if labels[l] == labels[i]]
        if len(similarTrajs) > 1:
            # for k in similarTrajs:
            #     A[i] += distMatrix[i,k]
            A[i] = sum(subDistMatrix[i, similarTrajs]) / (len(similarTrajs) - 1)
            otherClusters = list(set(labels) - set([labels[i]]))
            b = np.inf
            for j in otherClusters:
                # b = 0
                dissimilarTrajsJ = [l for l in range(len(labels)) if labels[l] == j]
                # for k in dissimilarTrajsJ:
                #     b += distMatrix[i,k]
                b = np.mean(subDistMatrix[i, dissimilarTrajsJ])
                if b < B[i]:
                    B[i] = b
                    argmins[i] = j
                # B[i,j] = np.mean(distMatrix[i,dissimilarTrajsJ])
            # argmins[i] = np.argmin(B[i])
            S[i] = (B[i] - A[i]) / max(B[i], A[i])
            if S[i] < 0:
                closestCluster[i] = argmins[i]

    return S, closestCluster, labels, subDistMatrix, shufSubDistMatrix

class EvalFuncs():
    def AvgSManual(self, X, odLabels, trajLabels, S):
        return np.mean(S)

    def PosSRatioManual(self, X, odLabels, trajLabels, S):
        return len(np.where(S > 0)[0]) / len(S)

    def ARI(self, X, odLabels, trajLabels, S):
        return sklearn.metrics.adjusted_rand_score(odLabels, trajLabels)

    def MI(self, X, odLabels, trajLabels, S):
        return sklearn.metrics.mutual_info_score(odLabels, trajLabels)

    def Homogeneity(self, X, odLabels, trajLabels, S):
        return sklearn.metrics.homogeneity_score(odLabels, trajLabels)

    def Completeness(self, X, odLabels, trajLabels, S):
        return sklearn.metrics.completeness_score(odLabels, trajLabels)

    def V(self, X, odLabels, trajLabels, S):
        return sklearn.metrics.v_measure_score(odLabels, trajLabels)

    def FMI(self, X, odLabels, trajLabels, S):
        return sklearn.metrics.fowlkes_mallows_score(odLabels, trajLabels)

    def S(self, X, odLabels, trajLabels, S):
        return sklearn.metrics.silhouette_score(X, trajLabels)

    def CHI(self, X, odLabels, trajLabels, S):
        return sklearn.metrics.calinski_harabasz_score(X, trajLabels)

    def DBI(self, X, odLabels, trajLabels, S):
        return sklearn.metrics.davies_bouldin_score(X, trajLabels)


def evaluate_trajectory(distMatrices, trajectories, odTrajLabels, refTrajIndices, nClusStart, nClusEnd,
                        clusRange=list(range(2, 15)), nIter=3, modelList=None, dataName="mot15", test=True,
                        evalNameList=None,
                        seed=0.860161037286291):
    t = Time('start')
    random.seed(seed)
    if modelList == None:
        modelList = [
            # (sklearn_extra.cluster.KMedoids(metric='precomputed', init='k-medoids++'), 'KMedoids'),
            (KMeans(), 'KMeans')
            # (sklearn.cluster.AgglomerativeClustering(affinity='precomputed', linkage='ward'), 'ward Agglo-Hierarch')
            , (AgglomerativeClustering(affinity='precomputed', linkage='complete'),
               'complete Agglo-Hierarch')
            , (AgglomerativeClustering(affinity='precomputed', linkage='average'),
               'average Agglo-Hierarch')
            , (AgglomerativeClustering(affinity='precomputed', linkage='single'),
               'single Agglo-Hierarch')
            # ,(sklearn.cluster.Birch(threshold=0.5, branching_factor=50), 'BIRCH')
            # (sklearn.mixture.GaussianMixture(covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=200, n_init=5, init_params='kmeans'), 'GMM')
            , (SpectralClustering(affinity='precomputed', n_jobs=-1), 'Spectral')
            , (OPTICS(metric='precomputed', n_jobs=-1), 'OPTICS')
            , (DBSCAN(metric='precomputed', n_jobs=-1), 'DBSCAN')
        ]

    cmap = list(colors.TABLEAU_COLORS)
    colormap = cmap
    repeat = len(modelList) // len(cmap)
    for k in range(repeat):
        colormap = colormap + cmap

    randArray = np.random.rand(nIter)  ###### distMatrices refTrajIndices, odTrajLabels
    tempDistMatrices = distMatrices
    if test:
        tempDistMatrices = [(distMatrix, distName) for (distMatrix, distName) in distMatrices if
                            distName in [dataName + "_GuLcssMatrix_param7", dataName + "_GuDtwMatrix_param-1",
                                         dataName + "_GuPfMatrix_param0.2"]]

    initEvalMatrix = np.zeros((len(tempDistMatrices), len(modelList), len(clusRange), len(randArray)))
    # print(initEvalMatrix)

    evalFuncs = EvalFuncs()

    if evalNameList == None:
        evalNameList = [methodName for (methodName, method) in
                        inspect.getmembers(evalFuncs, predicate=inspect.ismethod)]

    evalMeasures = []
    for evalName in evalNameList:
        matched = False
        for (methodName, method) in inspect.getmembers(evalFuncs, predicate=inspect.ismethod):
            if methodName == evalName:
                evalMeasures.append((initEvalMatrix.copy(), method, methodName))
                matched = True
                break
            # else:
            #     raise ValueError(f'{evalName} is not a valid name for an evaluation measure! It was skipped.')
            # cprint('{} is not a valid name for an evaluation measure! It was skipped.'.format(evalName), color='red', on_color='on_grey')
        if not matched:
            raise ValueError(f'{evalName} is not a valid name for an evaluation measure! It was skipped.')

    for idxMatrix, (distMatrix, distName) in enumerate(tempDistMatrices):
        t = Time(distName, color='yellow')

        for idxSeed, seed in enumerate(randArray):
            random.seed(a=seed)
            shufTrajectories = trajectories.copy()
            shufTrajIndices = list(range(distMatrix.shape[0]))
            # random.shuffle(shufTrajIndices)

            shufOdTrajLabels = np.zeros(odTrajLabels.shape)
            shufRefTrajIndices = [0 for _ in refTrajIndices]
            shufDistMatrix = np.zeros(distMatrix.shape)
            for i in range(len(shufTrajectories)):
                shufTrajectories[i] = trajectories[shufTrajIndices[i]]

            # startLabels, endLabels, startPoints, endPoints = OdClustering(modelList=None, nClusOriginSet=[11], nClusDestSet=[4], visualize=False, funcTrajectories=shufTrajectories)
            # refTrajIndices, odTrajLabels = OdMajorClusters(threshold=10)

            shufOdTrajLabels = odTrajLabels.copy()
            shufRefTrajIndices = refTrajIndices.copy()
            for i in range(len(shufTrajIndices)):
                shufOdTrajLabels[i] = odTrajLabels[shufTrajIndices[i]]
            for i in range(len(refTrajIndices)):
                shufRefTrajIndices[i] = np.where(shufTrajIndices == refTrajIndices[i])[0][0]
            for i in range(shufDistMatrix.shape[0]):
                for j in range(shufDistMatrix.shape[1]):
                    shufDistMatrix[i, j] = distMatrix[shufTrajIndices[i], shufTrajIndices[j]]

            for idxModel, (model, modelName) in enumerate(modelList):
                for idxClus, nClus in enumerate(clusRange):
                    # model1 = model.copy()
                    if modelName == 'DBSCAN':
                        if 'dtw' in distName:
                            model.eps = 3
                        elif 'lcss' in distName:
                            model.eps = 0.2
                        elif 'pf' in distName:
                            model.eps = 3
                        else:
                            cprint(
                                'Epsilon value not specified yet for {} algorithm in the code. default value 0.5 is used.'.format(
                                    distName), color='red', on_color='on_yellow')
                    model.n_clusters = nClus
                    S, closestCluster, labels, subDistMatrix, shufSubDistMatrix = Silhouette(model=model,
                                                                                             distMatrix=shufDistMatrix,
                                                                                             trajIndices=shufRefTrajIndices)
                    trajLabels = labels
                    for idxEval, (_, evalFunc, evalName) in enumerate(evalMeasures):
                        try:
                            evalMeasures[idxEval][0][idxMatrix, idxModel, idxClus, idxSeed] = evalFunc(
                                subDistMatrix, shufOdTrajLabels[shufRefTrajIndices], trajLabels, S)
                        except:
                            cprint(f'Evaluation metric {evalName} failed to work', color='red', on_color='on_grey')
                    # avgS[idxMatrix, idxModel, idxClus, idxSeed] = np.mean(S)
                    # posSIndices = np.where(S>0)[0]
                    # posSRatio[idxMatrix, idxModel, idxClus, idxSeed] = len(posSIndices)/len(S)
                    # ARI[idxMatrix, idxModel, idxClus, idxSeed] = round(sklearn.metrics.adjusted_rand_score(shufOdTrajLabels[shufRefTrajIndices], trajLabels), 3)
                # t = Time('{} model is done'.format(title))
            t = Time('idxSeed {} is done'.format(idxSeed))

    try:
        os.mkdir('./data/' + dataName + "_output")
    except:
        pass

    if not test:
        for (evalMatrix, evalFunc, evalName) in evalMeasures:
            pickle_out = open(
                f'./../output/mot15/{dataName}_O{str(nClusStart)}-D{str(nClusEnd)}_{evalName}.pickle', "wb")
            pickle.dump(evalMatrix, pickle_out)
            pickle_out.close()
            # pickle_out = open('./data/'+dataName+"_output/"+dataName+"_O"+str(nClusStart)+"-D"+str(nClusEnd)+"_ARI.pickle", "wb")
        # pickle.dump(ARI, pickle_out)
        # pickle_out.close()
        # pickle_out = open('./data/'+dataName+"_output/"+dataName+"_O"+str(nClusStart)+"-D"+str(nClusEnd)+"_avgS.pickle", "wb")
        # pickle.dump(avgS, pickle_out)
        # pickle_out.close()
        # pickle_out = open('./data/'+dataName+"_output/"+dataName+"_O"+str(nClusStart)+"-D"+str(nClusEnd)+"_posSRatio.pickle", "wb")
        # pickle.dump(posSRatio, pickle_out)
        # pickle_out.close()

    cprint('\n "tableResults.csv" not saved. Save it manually.\n', color='red', on_color='on_yellow')
    tableResults = []
    # tableResults=[['dataName', 'dist', 'distParam', 'algo', 'algoParam', 'nClus', 'iter', 'ARI', 'avgS', 'posSRatio']]
    # for idxDist, (s, alpha) in enumerate([('dtw', -1), ('lcss', 1), ('lcss', 2), ('lcss', 3), ('lcss', 5), ('lcss', 7), ('lcss', 10), ('pf', 0.1), ('pf', 0.2), ('pf', 0.3), ('pf', 0.4)]):
    #     for idxModel, (A, beta) in enumerate([('kmedoids', 'None'), ('kmeans', 'None'), ('agglo', 'complete'), ('agglo', 'average'), ('agglo', 'single'), ('spectral', 'None'), ('OPTICS', 'None'), ('DBSCAN', 'None')]):
    #         for idxK, k in enumerate(list(range(2,7))):
    #             for iter in (range(3)):
    #                 tableResults.append([dataName, s, alpha, A, beta, k, iter, ARI[idxDist][idxModel][idxK][iter], avgS[idxDist][idxModel][idxK][iter], posSRatio[idxDist][idxModel][idxK][iter]])
    # savetxt('./data/'+dataName+"_output/"+dataName+"_O"+str(nClusStart)+"-D"+str(nClusEnd)+'_tableResults.csv', X=tableResults, delimiter=',', fmt ='% s')

    for idxMatrix, (distMatrix, distName) in enumerate(tempDistMatrices):
        cprint(distName, color='green', on_color='on_grey')
        plt.figure(figsize=(24, 13))
        for idxEval, (evalMatrix, evalFunc, evalName) in enumerate(evalMeasures):
            fig = plt.subplot(len(evalMeasures) // 4 + 1, 4, idxEval + 1)
            for i in range(len(modelList)):
                fig.plot(clusRange, np.nanmean(evalMatrix[idxMatrix, i], axis=-1), color=colormap[i],
                         label=modelList[i][1])
                fig.fill_between(clusRange,
                                 np.nanmean(evalMatrix[idxMatrix, i], axis=-1) - np.nanstd(evalMatrix[idxMatrix, i],
                                                                                           axis=-1),
                                 np.nanmean(evalMatrix[idxMatrix, i], axis=-1) + np.nanstd(evalMatrix[idxMatrix, i],
                                                                                           axis=-1),
                                 color=colormap[i], alpha=0.3)  # , label=modelList[i][1])
            # fig.set_ylim(0,1)
            fig.set_xlim(0, max(clusRange))
            fig.legend(loc='lower left')
            fig.set_title(evalName)

        # fig = plt.subplot(1,3,2)
        # for i in range(len(modelList)):
        #     fig.plot(clusRange, np.nanmean(avgS[idxMatrix, i], axis=-1), color=colormap[i], label=modelList[i][1])
        #     fig.fill_between(clusRange, np.nanmean(avgS[idxMatrix, i], axis=-1)-np.nanstd(avgS[idxMatrix, i], axis=-1), np.nanmean(avgS[idxMatrix, i], axis=-1)+np.nanstd(avgS[idxMatrix, i], axis=-1), color=colormap[i], alpha=0.3)#, label=modelList[i][1])
        # fig.set_ylim(-1,1)
        # fig.set_xlim(0,max(clusRange))
        # fig.legend(loc='lower left')
        # fig.set_title("average silhouette values")

        # fig = plt.subplot(1,3,3)
        # for i in range(len(modelList)):
        #     fig.plot(clusRange, np.nanmean(posSRatio[idxMatrix, i], axis=-1), color=colormap[i], label=modelList[i][1])
        #     fig.fill_between(clusRange, np.nanmean(posSRatio[idxMatrix, i], axis=-1)-np.nanstd(posSRatio[idxMatrix, i], axis=-1), np.nanmean(posSRatio[idxMatrix, i], axis=-1)+np.nanstd(posSRatio[idxMatrix, i], axis=-1), alpha=0.3, color=colormap[i])#, label=modelList[i][1])
        # fig.set_ylim(0,1)
        # fig.set_xlim(0,max(clusRange))
        # fig.legend(loc='lower left')
        # fig.set_title("positive sil. value ratio")

        # plt.title(distName, color='w')
        plt.savefig('./../output/mot15/' + dataName + '_' + distName[:-4] + "_graphs.pdf", dpi=300)

        plt.show()

    return evalMeasures, tableResults

def clusterPlot(trajectories,model, distMatrix, trajIndices=None, S=np.array([]), closestCluster=np.array([]), title=None, plotTrajsTogether=False, plotTrajsSeperate=False, plotSilhouette=False, plotSilhouetteTogether=False, darkTheme=True,file_path=None):
    if darkTheme:
        tickColors = 'w'
    else:
        tickColors = 'black'

    if trajIndices==None:
        trajIndices = list(range(distMatrix.shape[0]))
    subDistMatrix = distMatrix[trajIndices][:,trajIndices]
    model = model.fit(subDistMatrix)
    labels = model.labels_
    if closestCluster==np.array([]):
        closestCluster = labels
    clusters = list(set(labels))
    minTrajX = min([min(tr[:, 0]) for tr in trajectories])
    minTrajY = min([min(tr[:, 1]) for tr in trajectories])
    maxTrajX = max([max(tr[:, 0]) for tr in trajectories])
    maxTrajY = max([max(tr[:, 1]) for tr in trajectories])

    try:
        nClus = model.n_clusters
    except:
        nClus = len(set(labels))
    cmap = list(colors.TABLEAU_COLORS)
    colormap = cmap
    repeat = nClus//len(cmap)
    for i in range(repeat):
        colormap = colormap + cmap

    if plotTrajsTogether:
        plt.figure(figsize=(16,12))
        for i, j in enumerate(trajIndices):
            tr = trajectories[j]
            plt.plot(tr[:,0], tr[:,1], c=colormap[labels[i]], linewidth=0.3)#, alpha=1)
            plt.scatter(tr[0,0], tr[0,1], c=100, s=2, marker='o')
        plt.xlim(minTrajX-20,maxTrajX+20)
        plt.ylim(minTrajY-20,maxTrajY+20)
        plt.tick_params(colors=tickColors)
        if title != None:
            plt.title(label=title, color=tickColors)

        plt.axis('off')
        plt.savefig(f'{file_path}/output/mot15/tra_cluster0.jpg')
        # plt.show()

    if plotTrajsSeperate:
        nRows = -(-nClus//4)
        plt.figure(figsize=(16,3*nRows), dpi=600)
        for i in range(nClus):
            fig = plt.subplot(nRows, 4, i+1)
            for j, k in enumerate(trajIndices):
                if labels[j] == clusters[i]:
                    tr = trajectories[k]
                    fig.plot(tr[:,0], tr[:,1], c=colormap[closestCluster[j]], linewidth=0.3)
                    fig.scatter(tr[0,0], tr[0,1], c=100, s=2, marker='o')
            fig.set_xlim(minTrajX-10,maxTrajX+10)
            fig.set_ylim(minTrajY-10,maxTrajY+10)
            fig.tick_params(colors=tickColors)
            fig.set_xticks([])
            fig.set_yticks([])
            plt.text(minTrajX+25,maxTrajY-20,str(i+1))
        if title != None:
            plt.suptitle(title, color=tickColors)

        plt.savefig(f'{file_path}/output/mot15/tra_cluster1.jpg')
        # plt.show()


    if plotSilhouette and S!=np.array([]):# and closestCluster!=np.array([]):
        cmap = list(colors.TABLEAU_COLORS)
        colormap = cmap
        repeat = nClus//len(cmap)
        for i in range(repeat):
            colormap = colormap + cmap
        nRows = -(-nClus//4)
        plt.figure(figsize=(16,3*nRows), dpi=600)
        for i in range(nClus):
            clusList = [j for j in range(len(labels)) if labels[j]==clusters[i]]
            sortedClusList = [clusList[i] for i in np.argsort(S[clusList])[::-1]]
            fig = plt.subplot(nRows, 4, i+1)
            # fig.bar(range(len(clusList)), S[clusList], color=colormap[i])
            fig.bar(range(len(sortedClusList)), S[sortedClusList], color=[colormap[closestCluster[j]] for j in sortedClusList])
            fig.set_ylim(-1,1)
            fig.tick_params(colors=tickColors)
            fig.set_xticks([])
            fig.set_yticks([])
            # plt.text(0.01*len(sortedClusList),0.8,str(i))
        if title != None:
            plt.suptitle(title, color=tickColors)
        plt.savefig(f'{file_path}/output/mot15/tra_cluster2.jpg')
        # plt.show()

    if plotSilhouetteTogether:
        sortedS = [i for i in np.argsort(S)[::-1]]
        plt.bar(range(len(sortedS)), S[sortedS], color=[colormap[closestCluster[j]] for j in sortedS])
        plt.ylim(-1,1)
        plt.tick_params(colors=tickColors)
        if title != None:
            plt.title(label=title, color=tickColors)
        plt.savefig(f'{file_path}/output/mot15/tra_cluster3.jpg')
        # plt.show()
    


    plotRootTrajectory = True
    clusterTra = []
    if plotRootTrajectory:
        # plt.axis('off')
        plt.figure(figsize=(16, 12))
        for i in range(nClus):

            tr = []
            for j, k in enumerate(trajIndices):
                if labels[j] == clusters[i]:
                    tr.append(trajectories[k])

            tr_mean = np.mean(np.array((tr)),axis=0)
            plt.plot(tr_mean[:, 0], tr_mean[:, 1], c=colormap[i], linewidth=0.3)
            clusterTra.append(tr_mean)
        plt.xlim(minTrajX - 10, maxTrajX + 10)
        plt.ylim(minTrajY - 10, maxTrajY + 10)
        plt.tick_params(colors=tickColors)
        plt.axis('off')
        plt.savefig(f'{file_path}/output/mot15/tra_cluster5.jpg')

        pickle_out = open(f'{file_path}/output/mot15/distances/trajectoryAfterCluster.pickle', "wb")
        pickle.dump(clusterTra, pickle_out)
        pickle_out.close()

    # if plotRootTrajectory:
    #     nRows = -(-nClus // 4)
    #     plt.figure(figsize=(16, 3 * nRows), dpi=600)
    #
    #     for i in range(nClus):
    #         fig = plt.subplot(nRows, 4, i + 1)
    #         tr = []
    #         for j, k in enumerate(trajIndices):
    #             if labels[j] == clusters[i]:
    #                 tr.append(trajectories[k])
    #
    #         tr_mean = np.mean(np.array((tr)),axis=0)
    #         fig.plot(tr_mean[:, 0], tr_mean[:, 1], c=colormap[closestCluster[j]], linewidth=0.3)
    #         fig.set_xlim(minTrajX - 10, maxTrajX + 10)
    #         fig.set_ylim(minTrajY - 10, maxTrajY + 10)
    #         fig.tick_params(colors=tickColors)
    #         fig.set_xticks([])
    #         fig.set_yticks([])
    #         plt.text(minTrajX - 5, maxTrajY + 5, str(i))
    #     plt.savefig('./../output/mot15/tra_cluster5.jpg')
