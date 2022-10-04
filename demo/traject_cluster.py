import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import sys

import torch
sys.path.insert(0,"./")
from utils.cluster_function import DistMatricesSection,OdClustering,OdMajorClusters,evaluate_trajectory,clusterPlot,Silhouette


def load_data():
    file_path = "./data/rough_trajectories.pt"
    data = torch.load(file_path)
    root = []
    for sample in data:
        tra = sample['root'][:60,:]
        tra = tra-tra[0,:]
        root.append(tra)
    print("trajectorres number: ",len(root))
    return root

def vis_trajectory(trajectory):
    for tra in trajectory:
        plt.plot(tra[:,0],tra[:,1])
    plt.savefig("trajectories.jpg")

def vis_goal(trajectory):
    goal_x = []
    goal_y = []
    for tra in trajectory:
        goal_x.append(tra[-1,0])
        goal_y.append(tra[-1,1])
    plt.scatter(goal_x,goal_y)
    plt.savefig("goal.jpg")

def evaluate():
    file = open("./../data/mot15distMatrices.pickle", 'rb')
    distMatrices = pickle.load(file)

    nClusDestSet = [4]
    endLabels, endPoints, nClusEnd = OdClustering(funcTrajectories=trajectories, nClusDestSet=nClusDestSet,
                                                  shuffle=False, nIter=1, visualize=False)

    startLabels = np.ones(537)
    nClusStart = 1
    refTrajIndices, odTrajLabels = OdMajorClusters(trajectories=trajectories, startLabels=startLabels,
                                                   endLabels=endLabels, threshold=10, visualize=True, test=True,
                                                   load=False)

    clusRange, nIter, test = list(range(2, 30)), 3, False
    evalMeasures, tableResults = evaluate_trajectory(clusRange=clusRange, nIter=nIter, test=test,
                                                     distMatrices=distMatrices,
                                                     trajectories=trajectories, odTrajLabels=odTrajLabels,
                                                     refTrajIndices=refTrajIndices, nClusStart=nClusStart,
                                                     nClusEnd=nClusEnd,
                                                     modelList=None, dataName="mot15")

def plot(trajectories):
    file_path = "./"
    file = open(f"{file_path}/output/mot15/distances/distMatrices.pickle", 'rb')
    distMatrices = pickle.load(file)
    for (distMatrix, f) in distMatrices:
        if f == "mot15_DtwMatrix_param-1":
            model = AgglomerativeClustering(affinity='precomputed', linkage='average')
            model.n_clusters = 28
            S, closestCluster, labels, subDistMatrix, shufSubDistMatrix = Silhouette(model=model,
                                                                                     distMatrix=distMatrix)  # , trajIndices=trajIndices)
            clusterPlot(trajectories,model, distMatrix, trajIndices=None, S=S, closestCluster=closestCluster, title=None,
                 plotTrajsTogether=True, plotTrajsSeperate=True, plotSilhouette=True, plotSilhouetteTogether=True,
                 darkTheme=False,file_path=file_path)

def d_cluster():
    distMatrices = DistMatricesSection(trajectories=trajectories,test=False)

    # _,_,_ = OdClustering(funcTrajectories=trajectories, shuffle=True, nIter=10,nClusDestSet=[i for i in range(1, 30)], visualize=False)

    # nClusDestSet = [4]
    # # modelNames=['KMedoids','KMeans','average Agglo-Hierarch','ward Agglo-Hierarch','BIRCH','GMM']
    # modelNames = ['average Agglo-Hierarch']
    # endLabels, endPoints, nClusEnd = OdClustering(funcTrajectories=trajectories, nClusDestSet=nClusDestSet,
    #                                               modelNames=modelNames, shuffle=False, nIter=1, visualize=True)
    #
    # startLabels = np.ones(537)
    # refTrajIndices, odTrajLabels = OdMajorClusters(trajectories=trajectories, startLabels=startLabels,
    #                                                endLabels=endLabels, threshold=10, visualize=True, test=True,
    #                                                load=False)

if __name__ == '__main__':
    trajectories = load_data()
    # vis_trajectory(trajectories)

    # d_cluster()
    # evaluate(trajectories)
    # plot(trajectories)

