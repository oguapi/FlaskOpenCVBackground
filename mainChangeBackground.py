import os

import cv2
import mediapipe as mp
import numpy as np

mp_selfie_segmentation= mp.solutions.selfie_segmentation #Call function to use

cap= cv2.VideoCapture(0, cv2.CAP_DSHOW)

bg_path= bg_path= os.path.join('.','data','background2.jfif')

with mp_selfie_segmentation.SelfieSegmentation(
    model_selection=0) as selfie_segmentation:
    while True:
        ret, frame= cap.read()
        if ret == False:
            break
        #Change the original format BGR to RGB
        frame_rgb= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #Applicant selfie segmentation
        result= selfie_segmentation.process(frame_rgb) #With this result we change the background

        #Simple thresholding (umbralizacion simple) to improve the mask
        _, th= cv2.threshold(result.segmentation_mask,0.75,255, cv2.THRESH_BINARY) #All 0.75 became 255 or white obtein a binary image
        th= th.astype(np.uint8) #Format with working opencv
        th= cv2.medianBlur(th,13) #apply a filter to soften the edges

        #Inverting the binary image to manage the background and after sum to the profile
        th_inv= cv2.bitwise_not(th)

        #Background
        bg_img= cv2.imread(bg_path)
        bg_img= cv2.resize(bg_img,(frame.shape[1],frame.shape[0]), interpolation= cv2.INTER_CUBIC)
        bg_img= cv2.GaussianBlur(bg_img, (15,15),0)
        bg= cv2.bitwise_and(bg_img,bg_img, mask= th_inv)

        #Foreground
        fg= cv2.bitwise_and(frame,frame, mask= th)

        # Background + Foreground
        output= cv2.add(bg, fg)

        cv2.imshow("Frame", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()