import numpy as np
import cv2 as cv
import subprocess
import time
import os
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
img_width, img_height = 150, 150



def detectFromVideo(videoFile): #function to read objects from video
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, input_shape = (150, 150, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Convolution2D(32, 3, 3, activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Flatten())
        model.add(Dense(output_dim = 128, activation = 'relu'))
        model.add(Dense(output_dim = 10, activation = 'softmax'))
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        model.load_weights('driver_state_detection_small_CNN.h5')
        print(model.summary())

        video = cv.VideoCapture(videoFile)
        count = 0
        if (video.isOpened()== False):
                print("Error opening video  file")
        while(video.isOpened()):
                ret, frame = video.read()
                if ret == True:
                        cv.imwrite("test.jpg",frame)
                        # cv.imwrite("img1/"+str(count)+".jpg",frame)
                        count=count + 1
                        imagetest = image.load_img("test.jpg", target_size = (150,150))
                        imagetest = image.img_to_array(imagetest)
                        imagetest = np.expand_dims(imagetest, axis = 0)
                        preds = model.predict_classes(imagetest)
                        print(preds)
                        text_label = "{}: {:4f}".format(str(preds), 80)
                        cv.putText(frame, text_label, (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv.imshow('Frame', frame)
                        if cv.waitKey(2500) & 0xFF == ord('q'):
                                break
                else:
                         break
        video.release()
        cv.destroyAllWindows()



if __name__ == '__main__':
        detectFromVideo('video.gif')
