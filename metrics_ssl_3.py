# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Thu Sep 29 14:54:43 2022

# @author: ozgu
# """

names = ['person', 'bicycle',
'car',
  'motorcycle',
  'airplane',
  'bus' ,
  'train ',
  'truck' ,
  'boat' ,
  'traffic light' ,
  'fire hydrant' ,
  'stop sign' ,
  'parking meter' ,
  'bench' ,
  'bird' ,
  'cat' ,
  'dog' ,
  'horse' ,
  'sheep' ,
  'cow' ,
  'elephant' ,
  'bear' ,
  'zebra' ,
  'giraffe' ,
  'backpack' ,
  'umbrella' ,
  'handbag' ,
  'tie',
  'suitcase',
  'frisbee',
  'skis',
  'snowboard',
  'sports ball',
  'kite',
  'baseball bat',
  'baseball glove',
  'skateboard',
  'surfboard',
  'tennis racket',
  'bottle',
  'wine glass',
  'cup',
  'fork',
  'knife',
  'spoon',
  'bowl',
  'banana',
  'apple',
  'sandwich',
  'orange',
  'broccoli',
  'carrot',
  'hot dog',
  'pizza',
  'donut',
  'cake',
  'chair',
  'couch',
  'potted plant',
  'bed',
  'dining table',
  'toilet',
  'tv',
  'laptop',
  'mouse',
  'remote',
  'keyboard',
  'cell phone',
  'microwave',
  'oven',
  'toaster',
  'sink',
  'refrigerator',
  'book',
  'clock',
  'vase',
  'scissors',
  'teddy bear',
  'hair drier',
  'toothbrush']



from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import average_precision_score

import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


import math


from sklearn.metrics import precision_score, recall_score, accuracy_score
y_true = [0, 0, 1, 1, 0, 0]
y_pred = [1, 1, 1, 1, 1, 1]


def get_p(actual, pred):
    #
    # true
    # print(f" (Precision) -- actual: {actual}--- pred: {pred}")
    inters =  intersection(actual, pred)
    ones_p =  find_1(pred)
     
    if ones_p == 0:
        return 0.0
    
    return round(float(inters)/float(ones_p), 3)



def find_1(data):
    count = 0
    
    for i in range(len(data)):
        
        if int(data[i]) == 1:
            count = count + 1
    
    return count 
 
def intersection(target, pred):
    
    count = 0
    if(len(pred) == 0 or len(target) == 0):
        return 0.0 
        
    for i in range(len(target)):
        if (int(target[i]) == 1) and (int(pred[i]) == 1):
            count = count + 1
    
    return count
    
    
# print(f"Precision: {get_p(y_true, y_pred)}")

def get_r(actual, pred):
    #
    # print(f"(recall)-- actual: {actual}--- pred: {pred}")
    inters =  intersection(actual, pred)
    ones_a =  find_1(actual)
    
    if ones_a == 0:
        return 0.0
    
    return round(float(inters)/float(ones_a), 3)

    
# print(f"Recall: {get_r(y_true, y_pred)}")
def get_acc(actual, pred):
    #
    total = 0
    if len(actual) == 0 or len(pred) == 0 :
        return 0.0
    
    inter = intersection(actual, pred)
    for i in range(len(pred)):
            
        if (pred[i] == 1) or (actual[i] == 1):
            total = total + 1
     
    if total == 0:
        return 0.0
    
    return round(float(inter)/float(total), 3)

    
def get_f1(target, pred):
    
    if get_p(target, pred) == 0.0 and  get_r(target, pred) == 0.0:
        return 0.0
    
    return round(float(2.0 * ((get_p(target, pred) * get_r(target, pred)) / (get_p(target, pred) + get_r(target, pred)))), 3)



# p = get_p(y_true, y_pred)
# r = get_r(y_true, y_pred)
# a = get_acc(y_true, y_pred)


# print(f"precision : {p}, recall : {r}, accuracy : {a},")



# array_act = []
# array_pred = []

# p_l1 = [1, 0, 1]

# p_l2 = [1, 1, 1]

# p_l3 = [1, 0, 1]

# p_l4 = [0, 0, 1]

# p_l5 = [0, 1, 0]

# p_l6 = [1, 1, 0]



# a_l1 = [1, 0, 1]

# a_l2 = [1, 0, 1]

# a_l3 = [1, 0, 1]

# a_l4 = [1, 0, 1]

# a_l5 = [1, 0, 1]

# a_l6 = [1, 1, 1]

# array_act.append(a_l1)
# array_act.append(a_l2)
# array_act.append(a_l3)
# array_act.append(a_l4)
# array_act.append(a_l5)
# array_act.append(a_l6)

# array_pred.append(p_l1)
# array_pred.append(p_l2)
# array_pred.append(p_l3)
# array_pred.append(p_l4)
# array_pred.append(p_l5)
# array_pred.append(p_l6)
import torch

def convert_one_zero(data, num_labels):

    updated_data = []
    # print(len(data))
    for i in range(len(data)):
        result = (data[i] >= 0.5)
        if result:
            updated_data.append(1)
        else:
            updated_data.append(0)
       
    updated_data = torch.tensor(updated_data)
    
    # print(f"len of data: {len(updated_data)}")
    # time.sleep(10)
    # print(f"Shape-Type :  {len(updated_data)}-{type(updated_data)}")
    # updated_data = torch.cat(updated_data, dim=0) 
    # return updated_data.reshape(len(data), num_labels)   
    return updated_data  
    
# print(array_act)
    
# print(len(array_act))

"""
index : class index
label_list_a : actual label list
label_list_p : predicted label list
"""
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support


def get_one_class_p(index, label_list_a, label_list_p):
  
    assert index >= 0 
    assert len(label_list_a) == len(label_list_p)
    
    list_p = []
    list_a = []
    # print(len(label_list_a))
    #number of samples
    for i in range(len(label_list_a)):
        # print(f"i-- {int(label_list_a[i][index])}")
       
        
        list_a.append(int(label_list_a[i][index]))
        list_p.append(int(label_list_p[i][index]))
        
    # print(f"pred: {list_p}" )
    # print(f"actual: {list_a}" )
    
    
    # print(f"precision of : {precision_score(list_a, list_p, average='weighted')}")
    # print(f"recall of : {recall_score(list_a, list_p, average='weighted')}")
    # print(f"accucary of : {accuracy_score(list_a, list_p)}")
    # print(f"f1-score of : {f1_score(list_a, list_p, average='weighted')}")
    # y_true = convert_one_zero(label_list_a, 80)
    # y_pred = convert_one_zero(label_list_p, 80)
    y_true = list_a
    y_pred = list_p
    
    p = precision_score(y_true, y_pred, average='binary')
    r = recall_score(y_true, y_pred, average='binary')
    # a = accuracy_score(y_true, y_pred)
    f = f1_score(list_a, list_p, average='binary')
    
    if math.isnan(f):
        f = 0.0
    
    ap = average_precision_score(y_true, y_pred)
    # print(f"ap(per label)-- {i}: {ap}")
    # print(p)
    return list_a, list_p, ap, p, r, f

# get_one_class_p(2, array_act, array_pred)


def get_overall_p_r_a_f(label_list_a, label_list_p):

    OP = []
    OR = []
    OF = []
    
    AP = []
    
    count_zero_pr = 0
    for i in range(len(label_list_a)):
        
        y_true = convert_one_zero(label_list_a[i], 80)
        y_pred = convert_one_zero(label_list_p[i], 80)
        
        # y_true = label_list_a[i]
        # y_pred = label_list_p[i]
        # y_pred = y_pred.round()
        
        # print(f"{i}: training -- prediction --1: {y_pred} \n")
        # print(f"{i}: training -- target --1 {y_true} \n\n")
        
        p = precision_score(y_true, y_pred, average='binary')
        r = recall_score(y_true, y_pred, average='binary')
        # a = accuracy_score(y_true, y_pred)
        f = (2 * p * r) / (p + r)
        
        if math.isnan(f):
            f = 0.0
        if p == 0.0:
            # print(f"{i}:pre -- {p} --- recall -- {r}")
            count_zero_pr += 1
            
        # print(f)
        # if p == 1.0:
            # print(f"{i}: precision: {p} -- recall: {r} -- accuracy: {a} --  f1_score: {f}")
        
        # print(classification_report(y_true, y_pred))
        ap = average_precision_score(y_true, y_pred)
        # print(f"ap - {i}:: {ap}")
        # OP = OP + p
        # OR = OR + r
        # OA = OA + a
        # OF = OF + f
        # precisions, recalls = precision_recall_curve(y_true=y_true, 
        #                                      pred_scores=pred_scores,
        #                                      thresholds=thresholds)
        OP.append(p)
        OR.append(r)
        # OA.append(a)
        OF.append(f)
        
        # AP = AP + ap
        AP.append(ap)
    print(f"Total number of zero p and r: {count_zero_pr}")
    return OP, OR, OF, AP
        
        

import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

# # actual = numpy.random.binomial(1,.9,size = 1000)
# # predicted = numpy.random.binomial(1,.9,size = 1000)

# confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

# cm_display.plot()
# plt.grid(False)
# plt.show() 
# # print(f"Precision: {get_p(y_true, y_pred)}")
with open('outputs_coco.txt', 'r') as pred_file:
    preds = pred_file.readlines()
    
# print(len(preds))
    
with open('actuals_coco.txt', 'r') as act_file:   
    targets = act_file.readlines()
    
# print(len(targets))
# print(len(targets[4]))

# for i in range(len(targets[0])):
#     print(targets[0][i])

# print(type(targets[0]))

#transform targets to expected form
one_label = [] 
target_list = [] 
# index = 0
for i in range(len(targets)):
    splitted_line = targets[i].split('\n')
    one_label = [] 
    # index = 0
    # print("\n")
    for j in range(len(splitted_line[0])):
        
        if splitted_line[0][j] != ',':
            # print(f"index:: {index} - {int(splitted_line[0][j])}")
            # index = index + 1
            one_label.append(int(splitted_line[0][j]))
           
            # print(splitted_line[0][j])
    # print(one_label)
    target_list.append(one_label)
   

# print(len(target_list))
# print((target_list[5]))


p_one_label = [] 
pred_list = [] 
 
for i in range(len(preds)):
    splitted_line = preds[i].split('\n')
    p_one_label = [] 
    for j in range(len(splitted_line[0])):
        
        if splitted_line[0][j] != ',':
            p_one_label.append(int(splitted_line[0][j]))
           
            # print(splitted_line[0][j])
   
    pred_list.append(p_one_label)
# print(len(target_list))
# print((pred_list[5]))

y_pred_list = []
y_true_list = []
num_labels = 80


# for a in range(len(pred_list)):
#     print(pred_list[a])
    
#     print(target_list[a])
#     print("\n")
    
overall_p_per = 0.0
overall_r_per = 0.0
overall_a_per = 0.0
overall_f_per = 0.0

AP = 0.0
# f, axes = plt.subplots(nrows=8, ncols=10, figsize=(30, 25))


# # # print(len(y_true_list))
# # # # print(len(y_pred_list))
# # print(multilabel_confusion_matrix(target_list, pred_list))

# index = 0
# for i in range(8):
#     for j in range(10):
#         # print("-----" + str(index))
#         # y_pred = classifier.fit(X_train, y_train).predict(X_test)
#         cf_matrix = metrics.confusion_matrix(y_true_list[index], y_pred_list[index])
#         disp = metrics.ConfusionMatrixDisplay(cf_matrix,
#                                       display_labels=[False, True])
       
       
#         disp.plot(ax=axes[i][j], xticks_rotation=0)
#         disp.ax_.grid(False)
        
#         disp.ax_.set_title(names[index])
       
#         # disp.ax_.set_title(key)
#         disp.im_.colorbar.remove()
#         # disp.ax_.set_xlabel('')
        
     
#         # if i!=0:
#         #     disp.ax_.set_ylabel('')
#         index = index + 1
# f.text(0.4, 0.1, 'Multi-label classification per label', ha='left')
# plt.subplots_adjust(wspace=0.95, hspace=0.85)


# f.colorbar(disp.im_, ax=axes)
# plt.savefig("conf_mat_per_label.png")

# plt.show() 
import heapq
def top_k_ap(list_obj, k):
    
    assert k>0
    
    total = 0.0
    
    arr = heapq.nlargest(k, list_obj)
    # print(len(arr))
    for i in range(len(arr)):
        total = total + arr[i]
        # print(f"{i}-- {arr[i]} -- total: {total}")
        
    
    return total
    
    
    
AP_label = []

with open('pred_per_label.txt', 'w+') as pf, open('rec_per_label.txt', 'w+') as rf, open('f1_per_label.txt', 'w+') as ff :
    for i in range(num_labels):
        y_true, y_pred, ap_f, p, r, f = get_one_class_p(i, target_list, pred_list)
        
        # print(f"len of y true: {len(y_true)} --- y_pred: {len(y_pred)}")
        y_pred_list.append(y_pred)
        y_true_list.append(y_true)
        
        # print(i)
        # print(precision_recall_fscore_support(y_true, y_pred, average=None, labels=names))
        
        # print(f"Label-{i}: precision: {p}, recall: {r}, accuracy: {a}, f1: {f} \n")
        
        overall_p_per = overall_p_per + p
        overall_r_per = overall_r_per + r
        # overall_a_per = overall_a_per + a
        overall_f_per = overall_f_per + f
        
        
        pf.write(str(p) + "\n")
        rf.write(str(r) + "\n")
        # af.write(str(a) + "\n")
        ff.write(str(f) + "\n")
        
        # AP = AP + ap_f 
        # AP_label.append(ap_f)


k = len(target_list)
# k = 20
print(f"Overall(per label): precision: {(overall_p_per/num_labels)}, recall: {(overall_r_per/num_labels)}, f1: {(overall_f_per/num_labels)} \n")

OP, OR, OF, sample_AP = get_overall_p_r_a_f(target_list, pred_list)
sample_mAP = top_k_ap(sample_AP, k)

op = top_k_ap(OP, k)
or_s = top_k_ap(OR, k)

of = top_k_ap(OF, k)

label_mAP = top_k_ap(AP_label, k)

number_samples = len(pred_list)
print(f"Overall(per sample) precision: {op/k} -- recall: {or_s/k} --  f1_score: {of/k}")

print(f"Mean average precision(mAP)-- Sample mAP: {(sample_mAP/k)}")

pf.close()
rf.close()

ff.close()

# f, axes = plt.subplots(nrows=8, ncols=10, figsize=(30, 25))


# # # print(len(y_true_list))
# # # # print(len(y_pred_list))
# # print(multilabel_confusion_matrix(target_list, pred_list))

# index = 0
# for i in range(8):
#     for j in range(10):
#         # print("-----" + str(index))
#         # y_pred = classifier.fit(X_train, y_train).predict(X_test)
#         cf_matrix = metrics.confusion_matrix(y_true_list[index], y_pred_list[index])
#         disp = metrics.ConfusionMatrixDisplay(cf_matrix,
#                                       display_labels=[False, True])
       
       
#         disp.plot(ax=axes[i][j], xticks_rotation=0)
#         disp.ax_.grid(False)
        
#         disp.ax_.set_title(names[index])
       
#         # disp.ax_.set_title(key)
#         disp.im_.colorbar.remove()
#         # disp.ax_.set_xlabel('')
        
     
#         # if i!=0:
#         #     disp.ax_.set_ylabel('')
#         index = index + 1
# f.text(0.4, 0.1, 'Multi-label classification per label', ha='left')
# plt.subplots_adjust(wspace=0.95, hspace=0.85)


# f.colorbar(disp.im_, ax=axes)
# plt.savefig("conf_mat_per_label_after_ssl.png")

# plt.show() 


y_pred_list.clear()
y_true_list.clear()




