from tkinter import *
import tkinter
import numpy as np
import imutils
#import dlib
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
import os
from keras.preprocessing import image
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import winsound


main = tkinter.Tk()
main.title("Video-Based Abnormal Driving Behavior Detection")
main.geometry("800x500")

global awgrd_model
global video

def loadModel():
    global awgrd_model
    img_width, img_height = 150, 150
    train_data_dir = 'dataset/imgs/train'
    validation_data_dir = 'dataset/imgs/validation'
    nb_train_samples = 22424
    nb_validation_samples = 1254
    nb_epoch = 10
    if os.path.exists('AWGRD_model.h5'):
        awgrd_model = Sequential()
        awgrd_model.add(Convolution2D(32, 3, 3, input_shape = (150, 150, 3), activation = 'relu'))
        awgrd_model.add(MaxPooling2D(pool_size = (2, 2)))
        awgrd_model.add(Convolution2D(32, 3, 3, activation = 'relu'))
        awgrd_model.add(MaxPooling2D(pool_size = (2, 2)))
        awgrd_model.add(Flatten())
        awgrd_model.add(Dense(output_dim = 128, activation = 'relu'))
        awgrd_model.add(Dense(output_dim = 10, activation = 'softmax'))
        awgrd_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        awgrd_model.load_weights('AWGRD_model.h5')
        print(awgrd_model.summary())
        pathlabel.config(text="          AWGRD Model Generated Successfully")
    else:
        awgrd_model = Sequential()
        awgrd_model.add(Convolution2D(32, 3, 3, input_shape = (150, 150, 3), activation = 'relu'))
        awgrd_model.add(MaxPooling2D(pool_size = (2, 2)))
        awgrd_model.add(Convolution2D(32, 3, 3, activation = 'relu'))
        awgrd_model.add(MaxPooling2D(pool_size = (2, 2)))
        awgrd_model.add(Flatten())
        awgrd_model.add(Dense(output_dim = 128, activation = 'relu'))
        awgrd_model.add(Dense(output_dim = 10, activation = 'softmax'))
        awgrd_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        train_datagen = ImageDataGenerator(rescale=1.0/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1.0/255)
        train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_height, img_width), batch_size=32, class_mode='categorical')
        validation_generator = train_datagen.flow_from_directory(validation_data_dir, target_size=(img_height, img_width), batch_size=32, class_mode='categorical')
        awgrd_model.fit_generator(train_generator, samples_per_epoch=nb_train_samples, nb_epoch=nb_epoch, validation_data=validation_generator, nb_val_samples=nb_validation_samples)
        awgrd_model.save_weights('driver_state_detection_small_CNN.h5')
        pathlabel.config(text="          AWGRD Model Generated Successfully")



def upload():
    global video
    filename = filedialog.askopenfilename(initialdir="Video")
    pathlabel.config(text="          Video loaded")
    video = cv.VideoCapture(filename)

def beep():
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)

def startMonitoring():
        while(True):
            ret, frame = video.read()
            print(ret)
            if ret == True:
                cv.imwrite("test.jpg",frame)
                imagetest = image.load_img("test.jpg", target_size = (150,150))
                imagetest = image.img_to_array(imagetest)
                imagetest = np.expand_dims(imagetest, axis = 0)
                predict = awgrd_model.predict_classes(imagetest)
                print(predict)
                msg = "";
                if str(predict[0]) == '0':
                        msg = 'Safe Driving'
                if str(predict[0]) == '1':
                        msg = 'Using/Talking Phone'
                        beep()
                if str(predict[0]) == '2':
                        msg = 'Talking On phone'
                        beep()
                if str(predict[0]) == '3':
                        msg = 'Using/Talking Phone'
                        beep()
                if str(predict[0]) == '4':
                        msg = 'Using/Talking Phone'
                        beep()
                if str(predict[0]) == '5':
                        msg = 'Drinking/Radio Operating'
                        beep()
                if str(predict[0]) == '6':
                        msg = 'Drinking/Radio Operating'
                        beep()
                if str(predict[0]) == '7':
                        msg = 'Reaching Behind'
                        beep()
                if str(predict[0]) == '8':
                        msg = 'Hair & Makeup'
                        beep()
                if str(predict[0]) == '9':
                        msg = 'Talking To Passenger'
                        beep()
                text_label = "{}: {:4f}".format(msg, 80)
                cv.putText(frame, text_label, (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv.imshow('Frame', frame)
                if cv.waitKey(2500) & 0xFF == ord('q'):
                   break
            else:
                break
        video.release()
        cv.destroyAllWindows()



def exit():
    global main
    main.destroy()


font = ('times', 25, 'bold')
title = Label(main, text='Video-Based Abnormal Driving Behavior Detection', anchor=W )
title.config(bg='#06113C', fg='#FF8C32')
title.config(font=font)
title.config(height=4, width=140)
title.place(x=0,y=0)


font1 = ('times', 15, 'bold')
loadButton = Button(main, text="Generate & Load AWGRD Model", command=loadModel)
loadButton.config(bg='#FF5F00', fg='#00092C')
loadButton.place(x=250,y=200)
loadButton.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='LIGHTBLUE', fg='white')
pathlabel.config(font=font1)
pathlabel.place(x=250,y=250)


uploadButton = Button(main, text="Upload Video", command=upload)
uploadButton.config(bg='#FF5F00', fg='#00092C')
uploadButton.place(x=250,y=300)
uploadButton.config(font=font1)

uploadButton = Button(main, text="Start Behaviour Monitoring", command=startMonitoring)
uploadButton.config(bg='#FF5F00', fg='#00092C')
uploadButton.place(x=250,y=350)
uploadButton.config(font=font1)

exitButton = Button(main, text="Exit", command=exit)
exitButton.config(bg='red', fg='white')
exitButton.place(x=250,y=400)
exitButton.config(font=font1)

#main.config(bg='chocolate1')
main.mainloop()
