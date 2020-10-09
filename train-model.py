from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import LeNet
from utils import BalancedDataGenerator
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2
import os
import pickle


data_dir = Path('data')
paths = list(data_dir.glob('*/*.png'))

data = []
labels = []

epochs = 50
batchSize = 28
# loop over input images
for imagePath in paths:
    # load image, preprocess, and store
    image = cv2.imread(str(imagePath))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = preprocess(image, 28, 28)
    image = cv2.resize(image, (28, 28))
    image = img_to_array(image)
    #print(image.shape)
    data.append(image)
    # get class label from file directory
    # root_directory/class_label/image_filename.jpg
    label = imagePath.parent.stem
    labels.append(label)

# normalize
data = np.array(data, dtype='float') / 255.0

print(data.shape)
labels = np.array(labels)
print(labels.shape)

print("[INFO] creating balanced dataset...")
# # train test split 80% train
trainX, testX, trainY, testY = train_test_split(data,
                                                labels, test_size=0.2,
                                                stratify=labels,
                                                random_state=42)
print(len(np.unique(trainY)))
print(len(np.unique(testY)))
assert len(np.unique(trainY)) == len(np.unique(testY))

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
class_names = lb.classes_
print(class_names)
trainY = lb.transform(trainY)
testY = lb.transform(testY)
# Serialize encoder.
with open('label-binarizer.pkl', 'wb') as f:
    pickle.dump((lb), f)

datagen = ImageDataGenerator(rotation_range=5,
                             width_shift_range=0.05,
                             height_shift_range=0.05,
                             brightness_range=(0.75,1.25),
                             horizontal_flip=False,
                             vertical_flip=False,
                             zoom_range=0.05)

train_bgen = BalancedDataGenerator(trainX, trainY, datagen, batch_size=batchSize)
val_bgen = BalancedDataGenerator(testX, testY, datagen, batch_size=batchSize)
train_steps_per_epoch = train_bgen.steps_per_epoch
val_steps_per_epoch = val_bgen.steps_per_epoch

# y_gen = [train_bgen.__getitem__(0)[1] for i in range(train_steps_per_epoch)]
# print(np.unique(y_gen, return_counts=True))

for X_batch, y_batch in train_bgen:
    # create a grid of 3x3 images
    tmp_labels = lb.inverse_transform(y_batch)
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
        plt.title(tmp_labels[i])
        plt.axis('off')
    # show the plot
    plt.show()
    break

# # # Initialize models
print("[INFO] compiling models...")
# checkpointer = ModelCheckpoint(r'models\best_model1.h5', monitor='val_accuracy', verbose=1, save_best_only=True,
#                                save_weights_only=False)
# callbacks = [checkpointer]
opt = Adam(lr=1e-3)
model = LeNet.build(width=28, height=28, depth=1, classes=len(class_names))
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(train_bgen,
              steps_per_epoch=train_steps_per_epoch,
              validation_data=val_bgen,
              validation_steps=val_steps_per_epoch,
              epochs=epochs, verbose=2)

print("[INFO] evaluating network...")
predictions = model.predict(testX)
# print(testY.argmax(axis=1))
# print(predictions.argmax(axis=1))
# print(class_names)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=class_names))

# print("[INFO] serializing network...")
# model.save('models')

# plot
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["accuracy"], label="acc")
plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
