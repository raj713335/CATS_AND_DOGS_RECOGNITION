import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dir_path = os.path.dirname(os.path.realpath("model_creator.py"))
dir_path=dir_path.replace("\\","/")

print(dir_path)



cats=dir_path+'/DATA/TRAIN/cats'
dogs = dir_path+'/DATA/TRAIN/dogs'




print('total training cats images:', len(os.listdir(cats)))
print('total training dogs images:', len(os.listdir(dogs)))



one_files = os.listdir(cats)
print(one_files[:])

two_files = os.listdir(dogs)
print(two_files[:])



import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_index = 2

next_one = [os.path.join(cats, fname)
             for fname in one_files[pic_index - 2:pic_index]]
next_two = [os.path.join(dogs, fname)
              for fname in two_files[pic_index - 2:pic_index]]

"""
for i, img_path in enumerate(next_one + next_two):
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('Off')
    plt.show()
    if i>10:
        break
"""

import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator


TRAINING_DIR=dir_path+'/DATA/TRAIN'
training_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')



VALIDATION_DIR = dir_path+"/DATA/VALIDATION"

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(640, 480),
    class_mode='categorical'
)
print(train_generator)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(640, 480),
    class_mode='categorical'
)



class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.8):
      print("\nReached 80% accuracy so cancelling training!")
      self.model.stop_training = True





callbacks = myCallback()




# FEEDBACK PYUISH USE BATCH NORMALIZATION, REDURE DROUPOUT to 1% and use DIALATION AND EROSION WITH CUT-OFFS 
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(640, 480, 2)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit_generator(train_generator, epochs=25, validation_data=validation_generator, verbose=1, callbacks=[callbacks])

model.save("Animal.h5")

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()

print("END")


