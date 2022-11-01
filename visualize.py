#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:06:21 2022

@author: ozgu
"""

from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random 

classLabels = ['person', 'bicycle','car','motorcycle','airplane','bus' ,'train ','truck' ,'boat' ,
'traffic light' ,'fire hydrant' ,'stop sign' ,'parking meter' ,'bench' ,'bird' ,'cat' ,'dog' ,'horse' ,
'sheep' ,'cow' ,'elephant' ,'bear' ,'zebra' ,'giraffe' ,'backpack' ,'umbrella' ,'handbag' ,'tie',
'suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet',
'tv','laptop','mouse', 'remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator',
'book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

f_list = []
features_list = []
with open('coco_after.txt', 'r') as pred_file:
    for line in pred_file:
        f_list.clear()
        for i in line.split(','):
            if i.strip():
               f_list.append(float(i))
               # print(float(i))
        features_list.append(f_list)
            
print(len(features_list))
print(features_list[0])


with open('coco_labels.txt', 'r') as act_file:   
    targets = act_file.readlines()
    

one_label = [] 
label_list = [] 
 
for i in range(len(targets)):
    splitted_line = targets[i].split('\n')
    one_label = [] 
    for j in range(len(splitted_line[0])):
        
        if splitted_line[0][j] != ',':
            one_label.append(int(splitted_line[0][j]))
           
            # print(splitted_line[0][j])
   
    label_list.append(one_label)
    
print(len(label_list))
print((label_list[0]))

def create_label(index, labels):
    
    assert index >= 0 
    assert len(labels) >0
    
    label = []
   

    for item_label in labels:
      # print(item_label)
      if item_label[index] == 1:
          label.append(1)
      else:
          label.append(0)
          
    return label
       
        
      
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("coco_corr.csv")


fig, ax = plt.subplots(figsize=(20,20)) 
sns.heatmap(df.iloc[: , 1:].corr(), vmax=1, vmin=-1, center=0, cmap='vlag', ax=ax)
plt.show()
    
    
    
# with open('actuals_coco.txt', 'r') as act_file:   
#     targets = act_file.readlines()

# #transform targets to expected form
# one_label = [] 
# target_list = [] 
# # index = 0
# for i in range(len(targets)):
#     splitted_line = targets[i].split('\n')
#     one_label = [] 
#     # index randomlist = = 0
#     # print("\n")
#     for j in range(len(splitted_line[0])):
        
#         if splitted_line[0][j] != ',':
#             # print(f"index:: {index} - {int(splitted_line[0][j])}")
#             # index = index + 1
#             one_label.append(int(splitted_line[0][j]))
           
#             # print(splitted_line[0][j])
#     # print(one_label)
#     target_list.append(one_label)
   

# print(len(target_list))
# print((target_list[5]))



# # junk data
# n_random_list = []
# rand_list=[]
# n=128
# total = 4000
# for index in range(total):
#     rand_list.clear()
#     for i in range(n):
#         rand_list.append(random.uniform(-50.5,50.0))
    
#     n_random_list.append(rand_list)
# print(len(n_random_list[7]))

# print((n_random_list[7]))


# index = 17
# cur_label = create_label(index, label_list)
# # print(cur_label)
# model = TSNE(n_components=2, random_state=0)

# tsne_data = model.fit_transform(features_list)



# tsne_data = np.vstack((tsne_data.T, cur_label)).T
# tsne_df = pd.DataFrame(data=tsne_data, columns=("d1", "d2", "label (presence/absence)" ))


# sns.FacetGrid(tsne_df, hue = "label (presence/absence)", size=16).map(plt.scatter, "d1", "d2").add_legend()
# plt.title(str(classLabels[index]))
# plt.show()



# from  sklearn import decomposition

# pca = decomposition.PCA()

# pca.n_components = 2

# pca_data = pca.fit_transform(features_list)

# print(pca_data.shape)


# pca_data = np.vstack((pca_data.T)).T
# pca_df = pd.DataFrame(data=pca_data, columns=("d1", "d2"))


# sns.FacetGrid(pca_df, size=16).map(plt.scatter, "d1", "d2").add_legend()
# plt.title("test")
# plt.show()
