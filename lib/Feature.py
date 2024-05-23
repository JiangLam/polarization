import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat, savemat
from sklearn.preprocessing import PolynomialFeatures


# pearson_co: calculate correlation to reduce pbps number.


def pearson_pbps(file_path, pearson=.9):
    pbp_name = ["FinalM14", "FinalM41", "FinalM44",
                "MMT_t1", "MMT_b", "MMT_b2", "MMT_A", "MMT_Abrio_R", "MMT_beta", "Bhls", "Bfs",
                "MMT_t_1213", "MMT_t_2131", "MMT_t_4243", "MMT_t_2434", "PDxcheng", "rqxcheng", "MMTPD", "MMTrq",
                "PLsubDL", "rLsubqL", "PLsubrL", "DLsubqL", "PDslcha", "rqslcha", "MM_Det", "MM_Norm", "MM_Trace",
                "P_vec", "D_vec", "P_dot_D", "LDOP",
                "MMPD_D", "MMPD_DELTA", "MMPD_delta", "MMPD_R",
                "MMCD_lambda1", "MMCD_lambda2", "MMCD_lambda3", "MMCD_lambda4", "MMCD_P1", "MMCD_P2", "MMCD_P3",
                "MMCD_PI", "MMCD_PD", "MMCD_S",
                "MMLD_D", "MMLD_delta", "MMLD_a22", "MMLD_a33", "MMLD_a44", "MMLD_aL", "MMLD_aLA"]

    pbps = loadmat(file_path + '/pbps.mat')['pbps']
    pbps_flat = pbps.reshape(pbps.shape[0]*pbps.shape[1], pbps.shape[2])

    Data = pd.DataFrame(pbps_flat, columns=pbp_name)
    corelation = np.array(Data.corr().abs())
    temp = []
    for i in range(np.size(corelation, 1) - 1):
        for j in range(i + 1, np.size(corelation, 1)):
            if corelation[i, j] > pearson:
                temp.append(j)
    temp = np.unique(temp)
    name = []
    for i in temp:
        name.append(pbp_name[i])
    for name in name:
        pbp_name.remove(name)
    print(pbp_name)
    print(np.shape(pbp_name))
    return pbp_name


def polar_graph(file_path, degree=2):
    groups = loadmat(file_path + '/groups.mat')['groups']
    print(np.shape(groups))
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
    graph = poly.fit_transform(groups)
    a = poly.get_feature_names_out()
    print(a)
    print(np.shape(graph))
    j, t = 0, 0
    #os.mkdir(file_path+'/polar graph')
    for i in range(1024):
        i = int(i)
        print("\r进度: {}% ".format(t), ">" * (t // 2), end="")
        j += 1
        t = int((j + 1) // 10.23)
        time.sleep(0.05)
        plt.imshow(graph[i, :].reshape(40, 52), vmin=0., vmax=1.)
        plt.xticks([]), plt.yticks([])
        plt.savefig(file_path + '/polar graph/' + str(i) + '.jpg')
    savemat(file_path + '/graph.mat', {'graph': graph})
    return graph


if __name__ == "__main__":
    fp = ['data/canceer/ROI2']
    for file_path in fp:
        data = polar_graph(file_path)

