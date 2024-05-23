import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat

# plot some result methods


def plot_cluster(file_path, label):
    m11 = loadmat(file_path+'/m11.mat')['CalibratedM11']
    km_labels = loadmat(file_path+'km_labels.mat')['km_labels']
    cluster_img = np.zeros((km_labels.shape[:2]))
    cn = km_labels.max() + 1
    for i in range(cn):
        cluster_img[km_labels == i] = label[i]
    fig, axs = plt.subplots(1, 2, sharey = True, figsize=(20, 20))
    axs[0].imshow(cluster_img, 'jet')
    axs[0].set_title('cluster result')
    fig.colorbar()
    axs[1].imshow(m11, 'gray')
    axs[1].set_title('m11')
    return cluster_img

