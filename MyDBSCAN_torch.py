# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 16:31:18 2020

@author: eee
"""

import numpy as np
import copy
import torch

class MyDBSCAN():
    def __init__(self, eps, minpts):
        super(MyDBSCAN, self).__init__
        self.eps=eps
        self.minpts=minpts
    
    def fit(self, input_data):
        self.counter=0
        self.data_len=len(input_data)
        self.data=[{'val':val, 'label':0} for val in copy.deepcopy(list(torch.tensor(input_data)))]
        self.data_matrix=copy.deepcopy(torch.tensor(input_data))
        current_label=1
        for idx, pt in enumerate(self.data):
            print('processing: '+str(idx)+' '+str(pt))
            if pt['label']!=0:
                continue
            neighbor_loc=self.regionQuery(pt, current_label)

            if len(neighbor_loc)>=self.minpts:
                print('Assign label '+str(current_label)+' to '+str(pt)+\
                      ' with '+str(len(neighbor_loc))+' neighbors')
                self.data[idx]['label']=current_label
                self.counter+=1
                print('label '+str(current_label)+': '+str(self.counter)+' points')
                self.growCluster(neighbor_loc, current_label)
            else:
                print(str(pt)+' is noise')
                self.data[idx]['label']=-1
            self.counter=0
            current_label+=1
    
    def regionQuery(self, chosen_pt, current_label):
        distance=torch.sum(torch.sqrt((chosen_pt['val']-self.data_matrix)**2), dim=1)
        neighbor_loc=list(torch.where(distance<=self.eps)[0])
        return neighbor_loc
    
    def growCluster(self, neighbor_locs, current_label):
        i = 0
        while (i<len(neighbor_locs)):
            idx=neighbor_locs[i]
            if self.data[idx]['label']==-1:
                print('Assign label '+str(current_label)+' to '+str(self.data[idx]))
                self.data[idx]['label']=current_label
                self.counter+=1
                print('label '+str(current_label)+': '+str(self.counter)+' points')
            elif self.data[idx]['label']==0:
                print('Assign label '+str(current_label)+' to '+str(self.data[idx]))
                self.data[idx]['label']=current_label
                self.counter+=1
                print('label '+str(current_label)+': '+str(self.counter)+' points')
                temp_neighbor_locs=self.regionQuery(self.data[idx], current_label)
                if len(temp_neighbor_locs)>=self.minpts:
                    neighbor_locs=neighbor_locs+temp_neighbor_locs
            i+=1

if __name__=="__main__":
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    
    # data load
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                                random_state=0)
    X = StandardScaler().fit_transform(X)
    ##########
    
    # mine
    cluster=MyDBSCAN(eps=0.3, minpts=10)
    cluster.fit(torch.tensor(X))
    results=cluster.data
    labels=np.array([results[i]['label'] for i in range(len(results))])
    ##########
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    ##########
    
    # Plot result
    import matplotlib.pyplot as plt
    
    plt.figure()
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
    
        class_member_mask = (labels == k)
    
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()