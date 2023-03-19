import cv2
import numpy as np
import os
def video2frames(file_path):
    # file_list = ["students003.avi","crowds_zara01.avi", "crowds_zara02.avi",  "eth.avi","hotel.avi"]
    file_list = ["hotel.avi"]
    img_list = ["students","zara1", "zara2","eth", "hotel"]
    img_path = "/data/eth_ucy/img/"
    for i in range(len(file_list)):
        file_name = file_path+file_list[i]
        cap = cv2.VideoCapture(file_name)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) -1
        img_file_path = img_path+img_list[i]
        count = 0
        print(f"start converting video {file_list[i]}")
        # os.makedirs(img_file_path)
        while cap.isOpened():
            # extract the frame
            ret,frame = cap.read()
            if not ret:
                continue
            cv2.imwrite(img_file_path+f"/{count+1}.jpg",frame)
            count = count+1  
            if (count > (video_length)):
                cap.release()
                print(f"Done extracting frames. {count} frames extracted.")   
                break   


if __name__ == '__main__':
    file_path = "/data/eth_ucy/data/"
    # crowds_zara01.avi students003.avi crowds_zara01.avi crowds_zara02.avi hotel.avi eth.avi
    video2frames(file_path)
