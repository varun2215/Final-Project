

#model.add(MaxPooling2D((2, 2), dim_ordering="th"))

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense

img_width, img_height = 150, 150

# the path to the training data
train_data_dir = 'dataset/imgs/train'
# the path to the validation data
validation_data_dir = 'dataset/imgs/validation'

# the number of training samples. We have 20924 training images, but actually we can set the
# number of training samples can be augmented to much more, for example 2*20924
nb_train_samples = 22424

# We actually have 1500 validation samples, which can be augmented to much more
nb_validation_samples = 1254

# number of epoches for training
nb_epoch = 10

# training s small convnet from scatch
# convnet: a simple stack of 3 convolution layer with ReLU activation and followed by a max-pooling layers
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape = (150, 150, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Convolution2D(32, 3, 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(output_dim = 128, activation = 'relu'))
model.add(Dense(output_dim = 10, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# augmentation configuration for training data
train_datagen = ImageDataGenerator(rescale=1.0/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# augmentation configuration for validation data (actually we did no augmentation to teh validation images)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# training data generator from folder
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_height, img_width),
                                                  batch_size=32, class_mode='categorical')

# validation data generator from folder
validation_generator = train_datagen.flow_from_directory(validation_data_dir, target_size=(img_height, img_width),
                                                       batch_size=32, class_mode='categorical')

# fit the model
model.fit_generator(train_generator, samples_per_epoch=nb_train_samples, nb_epoch=nb_epoch,
                    validation_data=validation_generator, nb_val_samples=nb_validation_samples)

# save the weights
model.save_weights('driver_state_detection_small_CNN.h5')
