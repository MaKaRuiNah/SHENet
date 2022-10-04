import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import json

import torch

def cal_top1_error(joint1):
    input_n  = 10
    output_n = 50
    n = len(joint1)
    fde10 = fde20 = fde30 = fde40 = fde50 = ade10 = ade20 = ade30 = ade40 = ade50 = 0
    for i in range(n):
        root_gt =  np.array(joint1[i]['root'])[input_n:input_n+output_n]   # 50*2
        root_pred = np.array(joint1[i]['pred_root']) # 50*2

        distances_pred = np.linalg.norm(root_pred-root_gt,axis=1)  # 50

        fde10 += distances_pred[9]
        fde20 += distances_pred[19]
        fde30 += distances_pred[29]
        fde40 += distances_pred[39]
        fde50 += distances_pred[49]

        ade10 += np.mean(distances_pred[:10])
        ade20 += np.mean(distances_pred[:20])
        ade30 += np.mean(distances_pred[:30])
        ade40 += np.mean(distances_pred[:40])
        ade50 += np.mean(distances_pred[:50])
    eval_summary = 'total error: ade10: %.2f  ade20: %.2f  ade30: %.2f  ade40: %.2f  ade50: %.2f\n' % (ade10/n,ade20/n,ade30/n,ade40/n,ade50/n)
    eval_summary += 'fde10: %.2f   fde20: %.2f   fde30: %.2f   fde40: %.2f   fde50: %.2f' %(fde10/n,fde20/n,fde30/n,fde40/n,fde50/n)
    print(eval_summary)

def cal_top3_error():
    input_n  = 10
    output_n = 50
    n = len(result)
    fde10 = fde20 = fde30 = fde40 = fde50 = ade10 = ade20 = ade30 = ade40 = ade50 = 0
    for i in range(n):
        root_gt =  np.array(result[i]['root'])[input_n:input_n+output_n]   # 50*2
        root_gt_rep = root_gt.reshape(1,output_n,2) .repeat(3,axis=0)
        root_pred = np.array(result[i]['pred_root']) # 3*50*2

        distances = np.linalg.norm(root_pred-root_gt,axis=2)  # 3*50

        mean_distances = np.mean(distances,axis=1)  #3

        idx = np.argmin(mean_distances)

        distances_pred = distances[idx]  #50

        fde10 += distances_pred[9]
        fde20 += distances_pred[19]
        fde30 += distances_pred[29]
        fde40 += distances_pred[39]
        fde50 += distances_pred[49]

        ade10 += np.mean(distances_pred[:10])
        ade20 += np.mean(distances_pred[:20])
        ade30 += np.mean(distances_pred[:30])
        ade40 += np.mean(distances_pred[:40])
        ade50 += np.mean(distances_pred[:50])
    eval_summary = 'total error: ade10: %.2f  ade20: %.2f  ade30: %.2f  ade40: %.2f  ade50: %.2f\n' % (ade10/n,ade20/n,ade30/n,ade40/n,ade50/n)
    eval_summary += 'fde10: %.2f   fde20: %.2f   fde30: %.2f   fde40: %.2f   fde50: %.2f' %(fde10/n,fde20/n,fde30/n,fde40/n,fde50/n)
    print(eval_summary)

if __name__ == "__main__":
    with open('./output/SHENet/PETS/outputs/res.json', 'r') as f:
        result = json.load(f)
    
    cal_top1_error(result)

        