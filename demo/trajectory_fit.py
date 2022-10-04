import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import sys
from numpy import polyfit,poly1d
from scipy.special import comb
import torch
sys.path.insert(0,"./")


def load_data():
    file_path = "./data/rough_trajectories.pt"
    # file = open(file_path, 'rb')
    # data = pickle.load(file)
    data = torch.load(file_path)
    root = []
    # print(data[0]['root'].shape)
    for sample in data:
        tra = sample['root'][:60,:]
        tra = tra-tra[0,:]
        root.append(tra)
    print("trajectorres number: ",len(root))
    return root

def get_bezier_parameters(X, Y, degree=3):
    def bpoly(n, t, k):
        """ Bernstein polynomial when a = 0 and b = 1. """
        return t ** k * (1 - t) ** (n - k) * comb(n, k)
        #return comb(n, i) * ( t**(n-i) ) * (1 - t)**i
    def bmatrix(T):
        """ Bernstein matrix for Bézier curves. """
        return np.matrix([[bpoly(degree, t, k) for k in range(degree + 1)] for t in T])
    def least_square_fit(points, M):
        M_ = np.linalg.pinv(M)
        return M_ * points
    T = np.linspace(0, 1, len(X))
    M = bmatrix(T)
    points = np.array(list(zip(X, Y)))
    
    final = least_square_fit(points, M).tolist()
    final[0] = [X[0], Y[0]]
    final[len(final)-1] = [X[len(X)-1], Y[len(Y)-1]]
    return np.array(final)

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=50):
    nPoints = len(points)

    # xPoints = np.array([p[0] for p in points])
    # yPoints = np.array([p[1] for p in points])
    xPoints = points[:,0]
    yPoints = points[:,1]
    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def curve_fit(x,y):
    coeff = polyfit(x,y,3)
    return poly1d(coeff)



def vis_trajectory(trajectory):
    fit_tra = []
    for tra in trajectory:
        x = tra[:,0]
        y = tra[:,1]
        f = curve_fit(x,y)
        fit_tra.append(np.array(f(x)))
        # plt.scatter(x,y,3)
        plt.plot(x,y)
        # plt.plot(x,f(x),linewidth =2)
    fit_tra  = np.array(fit_tra)
    trajectory = np.array(trajectory)
    error = np.sum(np.abs(fit_tra-trajectory[:,:,1]))
    print(error)
    plt.savefig("./test_img/trajectory_fit.jpg")

def vis_bezier_trajectory(trajectory):
    fit_tra = []
    for tra in trajectory:
        x = tra[:,0]
        y = tra[:,1]

        data = get_bezier_parameters(x, y, degree=2)
        # x_val = [x[0] for x in data]
        # y_val = [x[1] for x in data]
        xvals, yvals = bezier_curve(data, nTimes=1000)
        idx = torch.linspace(0,999,x.shape[0])
        idx = idx.int()
        y_preds = yvals[::-1][idx]
        fit_tra.append(y_preds)
        # print(np.sum(np.abs(y_preds-y)))
        # plt.scatter(x,y,3)
        plt.plot(xvals,yvals,linewidth=2)
        # plt.plot(x,y,linewidth=2)
        # plt.plot(x,f(x),linewidth =2)
    fit_tra  = np.array(fit_tra)
    trajectory = np.array(trajectory)
    error = np.sum(np.abs(fit_tra-trajectory[:,:,1]))/len(fit_tra)
    print(error)
    plt.savefig("./test_img/test/bezier1-3.jpg")

if __name__ == '__main__':
    data = load_data()
    # vis_trajectory(data[:100])
    vis_bezier_trajectory(data[:50])