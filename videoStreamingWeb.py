import os

from flask import Flask
from flask import render_template
from flask import Response
import cv2

import cv2
import mediapipe as mp
import numpy as np

app= Flask(__name__) #Initialize application, we use "_" because with that flask know if we ejecute 
                     # this strip from principal file or it is import

mp_selfie_segmentation= mp.solutions.selfie_segmentation #Call function to use

cap= cv2.VideoCapture(0, cv2.CAP_DSHOW)

bg_path= bg_path= os.path.join('.','data','background2.jfif')

def generate():
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
            th= th.astype(np.uint8)                         #Format with working opencv
            th= cv2.medianBlur(th,13)                       #apply a filter to soften the edges

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

            (flag, encodedImage)= cv2.imencode(".jpg", output) #This function compresses the image and
                                                            # stores it in the memory buffer, it will
                                                            # be encoded in 'jpg' to reduce the load.
            if not flag:
                continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                  bytearray(encodedImage) + b'\r\n') #It will allow us to generate each encoded frame
                                                     # as a byte array to be consumed by the browser

@app.route("/") #Call route decorator to say to flask what URL activate, in this case is principal
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype= "multipart/x-mixed-replace; boundary=frame") #Each frame replace the other

if __name__ == "__main__":
    app.run(debug=False)

cap.release()