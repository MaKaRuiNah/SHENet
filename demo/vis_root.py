from cv2 import error
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import json


def vis_img(img,root_pred,root_gt,save_path):
    n = root_gt.shape[0]
    m = root_pred.shape[0]
    input_n = 10
    for i in range(n):
        if i< input_n:
            gt_point = root_gt[i,:]
            cv2.circle(img, (int(gt_point[0]), int(gt_point[1])), 4, (255, 0, 0), 2)
        else:
            gt_point = root_gt[i,:]
            cv2.circle(img, (int(gt_point[0]), int(gt_point[1])), 4, (0, 255, 0), 2)

    for i in range(m):
        pred_point = root_pred[i,:]
        cv2.circle(img, (int(pred_point[0]), int(pred_point[1])), 4, (0, 0, 255), 2)

    cv2.putText(img, "pred trajectory", (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.putText(img, "gt trajectory", (10,100 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imwrite(save_path,img)


def run(joint,idx,input_n,output_n,save_path):
    root_pred = np.array(joint[idx]['pred_root'])
    root_gt =  np.array(joint[idx]['root'])
    img_path = joint[idx]['img_path']
    
    img = cv2.imread(img_path)
    vis_img(img.copy(),root_pred,root_gt[:input_n+output_n],f"{save_path}/{idx}.jpg")


if __name__ == "__main__":
    # lstm offsets classifys_offsets  scene1_curve, lstm_curve

    with open('./output/SHENet/PETS/outputs/res.json', 'r') as f:
        data = json.load(f)

    root_gt = []
    save_path = "./test_img/mot_curve/"

    input_n = 10
    output_n = 50
    for idx in range(1,len(data),5):
        run(data,idx,input_n,output_n,save_path)
