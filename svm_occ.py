#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:08:10 2022

@author: ozgu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
#from sklearn.inspection import DecisionBoundaryDisplay

# from sklearn.manifold import TSNE
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import random 


# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# from sklearn.svm import SVC # "Support vector classifier"
# from sklearn.datasets import make_blobs

from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import f1_score

# from sklearn.model_selection import GridSearchCV
# from sklearn import preprocessing
import warnings

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error

# Import math Library
import math 

# K-fold cross validation
from sklearn.model_selection import cross_val_score

# Preserving the percentage of samples for each class
from sklearn.model_selection import StratifiedKFold

# One-class SVM classification 
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore")

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
counter = 0
index_plot = 4 # Order for evaluation 
m = "occ" #Method name for files

with open('info_batch_epoch0_iter4800.txt', 'r') as pred_file:
    Lines = pred_file.readlines()
    for line in Lines:
        f_list.clear()
        for i in line.split(','):
            if i.strip():
               f_list.append(float(i))
               # print(float(i))
        # counter = counter + 1
        # print(counter)
        # print(f_list)
        features_list.append(f_list.copy())
        
            
# print(len(features_list))
# print(features_list[0])
pred_file.close()


one_label = [] 
label_list = [] 
with open('info_labels0_epoch4800_iter.txt', 'r') as act_file:   
    targets = act_file.readlines()
    for i in range(len(targets)):
        splitted_line = targets[i].split('\n')
        one_label.clear()
        for j in range(len(splitted_line[0])):
            
            if splitted_line[0][j] != ',':
                one_label.append(int(splitted_line[0][j]))
               
                # print(splitted_line[0][j])
       
        label_list.append(one_label.copy())


    
# print(len(label_list))
# print((label_list[0]))
act_file.close()

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
       
def plot_roc_curve(true_y, y_prob, class_name, auc, i, method="linearSVM"):
    """
    plots the roc curve based of the probabilities
    """
    plt.clf()
    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    # print(f"fpr : {fpr} -- tpr : {tpr}")
    plt.plot(fpr, tpr, color="orange", label="ROC curve (area = %0.2f)" % auc)
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.title(class_name)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate') 
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(method + class_name + "_" + str(i) + ".png") 
      
     
        
     
# for i in range(5):
#     print(f"{i}-- {features_list[i]}")
#     print(label_list[i])

# correlation matrix for labels
# import seaborn as sns
# import matplotlib.pyplot as plt
# df = pd.read_csv("coco_corr.csv")
# fig, ax = plt.subplots(figsize=(20,20)) 
# sns.heatmap(df.iloc[: , 1:].corr(), vmax=1, vmin=-1, center=0, cmap='vlag', ax=ax)
# plt.show()

#for one class  
# index = 1
# cur_label = create_label(index, label_list)

for index in range(len(classLabels)):
    
    cur_label = create_label(index, label_list)
    cur_label = np.array(cur_label, dtype=np.int64)
    # print(len(cur_label))
# model = TSNE(n_components=2, random_state=0)

# features_list = np.array(features_list, dtype=np.float64)
# tsne_data = model.fit_transform(features_list)

# print("t-SNE is done!")
# tsne_data_stacked = np.vstack((tsne_data.T, cur_label)).T
# tsne_df = pd.DataFrame(data=tsne_data_stacked, columns=("d1", "d2", str(classLabels[index]) +"(presence/absence)" ))

# sns.FacetGrid(tsne_df, hue = str(classLabels[index]) + "(presence/absence)", size=16 ).map(plt.scatter, "d1", "d2").add_legend()
# plt.savefig(str(classLabels[index]) + ".png")
# # X, y = make_blobs(n_samples=40, centers=2, random_state=0)

# # normalization
# # X = (tsne_data - tsne_data.min())/ (tsne_data.max() - tsne_data.min())



# X = tsne_data
    with open("occ_svm_coco_roc.txt", "a+") as file_roc :
        # f.write(str(classLabels[index]) + ",")
        # file_acc.write(str(classLabels[index]) + ",")
        file_roc.write(str(classLabels[index]) + ",")
        
        X = np.array(features_list)
        y = cur_label
        
        unique, counts = np.unique(y, return_counts=True)
      
        if (len(unique) != 2) or (counts[0] < 2 or counts[1] < 2):
            print("Not enough samples!!!")
            # f.write(str(0.000) + ",")
            # file_acc.write(str(0.000) + ",")
            file_roc.write(str(0.000) + ",")
            # f.write("\n")
            # file_acc.write("\n")
            file_roc.write("\n")
            continue
        else: 
        # else:
            
        #     X_train, X_test, y_train, y_test = train_test_split(X, y,
        #                                         test_size=0.2, stratify=y,
        #                                         random_state=32)# 
            
            # plt.figure(figsize=(15, 10))
            
            
            print(classLabels[index])
            # print("The content:", counts[1])
            for i, C in enumerate([100]):
                
                # "hinge" is the standard SVM loss
                # clf = LinearSVC(C=C, loss="hinge", random_state=42).fit(X, y)
                
                # define cross-validation method to use
                # scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
                
                
                # classifier = KNeighborsClassifier(n_neighbors=2)
                # classifier.fit(X_train, y_train)
                # y_pred = classifier.predict(X_test)
                # print(confusion_matrix(y_test, y_pred))
                # print(classification_report(y_test, y_pred))
                # auc = roc_auc_score(y_test, y_pred)
                # print(f'AUC (knn) score: {auc}') 


                # clf = svm.SVC(kernel='linear', C=C, gamma='auto').fit(X, y)
                # obtain the support vectors through the decision function
                # decision_function = clf.decision_function(X_train)
                # print(decision_function)
                # we can also calculate the decision function manually
                # decision_function = np.dot(X, clf.coef_[0]) + clf.intercept_[0]
               
                # The support vectors are the samples that lie within the margin
                # boundaries, whose size is conventionally constrained to 1
                # support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
                # print(type(support_vector_indices))
                # support_vectors = X_train[support_vector_indices]
                # print(f"number of support vectors= {len(support_vectors)}")
            
                # predictions_poly = clf.predict(X_test)
               
                # predictions_poly = predictions_poly.tolist()
                #print(f"pred: {len(predictions_poly)}")
                #print(f"actual: {len(y_test)}")
                # accuracy_poly = accuracy_score(y_test, predictions_poly)
                # f1_poly = f1_score(y_test, predictions_poly)
                
                # auc = roc_auc_score(y_test, predictions_poly)
                # print("%0.2f AUC score with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
                # # print(f'AUC score: {auc}') 
                # # print(f'Error (SVM-Linear): {math.dist(y_test, predictions_poly)}')
                    
                # # plot_roc_curve(y_test, predictions_poly, str(classLabels[index]), auc, index, m)
                
                # # print(f"{i} - acc: {accuracy_poly} -- f1: {f1_poly}")
                # # f.write(str(f1_poly) + ",")
                # # file_acc.write(str(accuracy_poly) + ",")
                # file_roc.write(str(scores.mean()) + ",")
                
                skf = StratifiedKFold(n_splits=5)
                auc = 0.0
                auc_list = []
                error = 0.0 
                min_val = 0.0
                for train_index, test_index in skf.split(X, y):
                    # print("TRAIN:", train_index, "TEST:", test_index)
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    
                    # print("TRAIN:", len(X_train), "TEST:", len(X_test))
                    
                    #unsupervised classification
                    clf = OneClassSVM(kernel = 'linear').fit(X_train)
                    
                    predictions_poly = clf.predict(X_test)
                    # predictions_poly = predictions_poly.tolist()
                    
                    # -1 for outliers and 1 for inliers
                    # -1 represents the 0 for our situation
                    predictions_poly = [0 if x==-1 else 1 for x in predictions_poly]
                    predictions_poly = np.asarray(predictions_poly)
                   
                    auc += roc_auc_score(y_test, predictions_poly)
                    
                    if roc_auc_score(y_test, predictions_poly) > min_val:
                        #Update
                        min_val = roc_auc_score(y_test, predictions_poly)
                        y_pred_max = predictions_poly
                        y_test_max = y_test
                      
                    
                    
                    
                    
                    auc_list.append(roc_auc_score(y_test, predictions_poly))
                    error += mean_squared_error(y_test, predictions_poly)
                    
                    # print(f"error:  {error} -- y_pred: {(predictions_poly)} -- y_test: {y_test} \n")
                    # auc = auc // 5
                    # print("%0.2f AUC score " % (auc))
                 
                # print(len(auc_list))
                auc = auc / 5.0
                # print("%0.2f AUC score, standard deviation %0.3f" % (auc, (np.asarray(auc_list)).std() ))
                error = error / 5.0
                # print("%0.2f Error " % (error))
                file_roc.write(str(auc) + ",std,"  + str((np.asarray(auc_list)).std()) + ",error," + str(error) + "\n")
                auc_list.clear()
                
                # Best score's ROC and AUC value
                plot_roc_curve(y_test_max, y_pred_max, str(classLabels[index]), min_val, index_plot, m)
                
                
            # f.write("\n")
            # file_acc.write("\n")
          
        #     plt.subplot(1, 3, i + 1)
        #     plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired )
            
        #     ax = plt.gca()
        # f.close()
        # file_acc.close()
        file_roc.close()
#     DecisionBoundaryDisplay.from_estimator(
#         clf,
#         X,
#         ax=ax,
#         grid_resolution=50,
#         plot_method="contour",
#         colors="k",
#         levels=[-1, 0, 1],
#         alpha=0.5,
#         linestyles=["--", "-", "--"],
       
#     )
#     plt.scatter(
#         support_vectors[:, 0],
#         support_vectors[:, 1],
#         s=100,
#         linewidth=1,
#         facecolors="none",
#         edgecolors="k",
#     )
#     plt.title("C=" + str(C))
# plt.tight_layout()
# plt.savefig(str(classLabels[index])+ "_svm" + ".png")
# plt.show()


