from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Activation, Dropout
from skimage.measure import regionprops, label
import matplotlib.patches as mpatches
from skimage.color import label2rgb
from PIL import ImageDraw, Image
import matplotlib.pyplot as plt
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
import os
from uiautomator import device as d
from time import sleep

# Part - 1


def image_pr():

# Take a screenshot using UI-Automator
user_input = 'Shutter'
# d.screenshot("D:\multiclass\Input_image\samsung.jpg")
# sleep(2)
vimage = cv2.imread("D:\multiclass\Input_image\samsung.jpg")
# cv2.imshow("Image", vimage)

# Convert the image to grayscale
gray = cv2.cvtColor(vimage, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Image", gray)

# Extract the edges
edged = cv2.Canny(gray, 30, 150)
# cv2.imshow("Image", edged)
# cv2.waitKey(0)

# Label the edges
label_image = label(edged)
image_label_overlay = label2rgb(label_image, image=gray)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(image_label_overlay)
list1 = []

# Draw the rectangle
for region in regionprops(label_image):
if region.area >= 40:
# Coordinates in form of y x height width
minr, minc, maxr, maxc = region.bbox
rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='yellow',
linewidth=2)
list1.append((minc, minr, maxc-minc, maxr-minr))
ax.add_patch(rect)

ax.set_axis_off()
plt.tight_layout()
# plt.show()

# Newlist having coordinates in form of x1,y1,x2,y2
list2 = []
length = len(list1)
for i in range(length):
list2.append([int(list1[i][0]), int(list1[i][1]), int(list1[i][0]+list1[i][2]),
int(list1[i][1]+list1[i][3])])

# Code to remove the overlapping of rectangle
temp_list = []
for index, (px1, py1, px2, py2) in enumerate(list2):
for cx1, cy1, cx2, cy2 in list2[index+1:]:
if cx1 >= px1 and cy1 >= py1 and cx2 <= px2 and cy2 <= py2:
temp_list.append([cx1, cy1, cx2, cy2])

for bounds in temp_list:
if bounds in list2:
list2.remove(bounds)

file = 'D:\multiclass\Input_image\samsung.jpg'
input_img = Image.open(file)
draw = ImageDraw.Draw(input_img)
vdir = "D:\multiclass\cropped_new"
filelist = [f for f in os.listdir(vdir) if f.endswith(".jpg")]
for f in filelist:
os.remove(os.path.join(vdir, f))

# Draw the rectangles
i = 0
for bounds in list2:
f = os.path.join(vdir, "cropped" + str(i) + ".jpg")
# draw.rectangle(bounds, width=2)
crop_file = input_img.crop(bounds)
crop_file.save(f, "PNG")
i = i+1
input_img.save('D:\multiclass\modified_image\samsung_modified.jpg', "PNG")

return draw, user_input, list2, input_img

# Part - 2


def model(draw, user_input, list2, input_img):

# Creating sequential model
classifier = Sequential()

# Multiple CNN layers
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten Layer to get a vector connect all the layers with the hidden layers
classifier.add(Flatten())
# Fully connected layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=5, activation='softmax'))

# Compiling the model
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Part - 3 (Train the model)

# Code now to load our images in our model frst we do something to avoid overfitting
# train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.1, zoom_range = 0.2,horizontal_flip = True)
# test_datagen = ImageDataGenerator(rescale = 1./255)
# training_set = train_datagen.flow_from_directory('D:\multiclass\dataset\\train',
# target_size = (64, 64),batch_size = 32, class_mode = 'categorical')
# test_set = test_datagen.flow_from_directory('D:\multiclass\dataset\\validation',
# target_size = (64, 64),batch_size = 32, class_mode = 'categorical')
#
# Fitting data into model
# history = classifier.fit_generator(training_set, steps_per_epoch=30,epochs=39,
# validation_data=test_set, validation_steps=9, verbose=1)
#
# Plotting
# plt.figure(figsize=[8,6])
# plt.plot(history.history['loss'],'r',linewidth=3.0,label="loss")
# plt.plot(history.history['val_loss'],'b',linewidth=3.0,label="val_loss")
# plt.xlabel('epoch',fontsize=16)
# plt.ylabel('loss',fontsize=16)
# plt.title('loss curve' ,fontsize=16)
# plt.legend()
# plt.show()
#
# plt.figure(figsize = [8,6])
# plt.plot(history.history['acc'],'r',linewidth = 3.0,label="accuracy")
# plt.plot(history.history['val_acc'],'b',linewidth = 3.0,label="val_accuracy")
# plt.xlabel('epoch',fontsize=16)
# plt.ylabel('acc',fontsize=16)
# plt.title('acc curve',fontsize=16)
# plt.legend()
# plt.show()
classifier.load_weights('D:\multiclass\weight_file\multiclass1.h5')

# Part -4 (Predicting result)

# Python dictionary to store the labels
label_dict = {
0: 'Flash',
1: 'Settings',
2: 'Shutter',
3: 'Swap',
4: 'wrong'
}

files = os.listdir('D:\multiclass\cropped_new')
list3 = []

for file in files:
test_image = image.load_img(os.path.join('D:\multiclass\cropped_new', file), target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
data = test_image.astype('float32')
data /= 255
result = classifier.predict(data)
maximum = np.amax(result)
t = result.argmax()
if maximum >= 0.8 and t < 4:
if user_input == label_dict[t]:
print("Predicted Category : {}".format(label_dict[t]), "file name:", file, "with accuracy:",
maximum)
r = int(file[7:-4])
# d.click((newlist[r][0] + newlist[r][2]) / 2, (newlist[r][1] + newlist[r][3]) / 2)
# print(newlist[r][0], newlist[r][1], newlist[r][2], newlist[r][3])
list3.append([int(list2[r][0]), int(list2[r][1]), int(list2[r][2]), int(list2[r][3])])
sleep(6)

# Highlighting the result
for bounds in list3:
draw.rectangle(bounds, outline="#F8F806")
print("Image Modified!")
input_img.save('D:\multiclass\modified_image\samsung_modified.jpg', "PNG")

# Main function


def main():
draw, user_input, list2, input_img = image_pr()
model(draw, user_input, list2, input_img)


main()










from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Activation, Dropout
from skimage.measure import regionprops, label
import matplotlib.patches as mpatches
from skimage.color import label2rgb
from PIL import ImageDraw, Image
import matplotlib.pyplot as plt
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
import os
from uiautomator import device as d
from time import sleep

# Part - 1


def image_pro():
d.screenshot("D:\Basket_binary_classification\input_img\\basket (18).jpg")
sleep(2)
vimage = cv2.imread("D:\Basket_binary_classification\input_img\\basket (18).jpg")
# cv2.imshow("Image", vimage)

# Convert the image to grayscale
gray = cv2.cvtColor(vimage, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Image", gray)

# Extract the edges
edged = cv2.Canny(gray, 30, 150)
# cv2.imshow("Image", edged)
# cv2.waitKey(0)

# Label the edges
label_image = label(edged)
image_label_overlay = label2rgb(label_image, image=gray)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(image_label_overlay)
list1 = []

# Draw the rectangle
for region in regionprops(label_image):
if region.area >= 40:
# Coordinates in form of y x height width
minr, minc, maxr, maxc = region.bbox
rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='yellow',
linewidth=2)
list1.append((minc, minr, maxc-minc, maxr-minr))
ax.add_patch(rect)

ax.set_axis_off()
plt.tight_layout()
# plt.show()

# List2 having coordinates in form of x1,y1,x2,y2
list2 = []
length = len(list1)
for i in range(length):
list2.append([int(list1[i][0]), int(list1[i][1]), int(list1[i][0]+list1[i][2]),
int(list1[i][1]+list1[i][3])])

# Code to remove the overlapping of rectangle
temp_list = []
for index, (px1, py1, px2, py2) in enumerate(list2):
for cx1, cy1, cx2, cy2 in list2[index+1:]:
if cx1 >= px1 and cy1 >= py1 and cx2 <= px2 and cy2 <= py2:
temp_list.append([cx1, cy1, cx2, cy2])

for bounds in temp_list:
if bounds in list2:
list2.remove(bounds)

file = 'D:\Basket_binary_classification\input_img\\basket (18).jpg'
img = Image.open(file)
draw = ImageDraw.Draw(img)
vdir = "D:\Basket_binary_classification\crop_img"
filelist = [f for f in os.listdir(vdir) if f.endswith(".jpg")]
for f in filelist:
os.remove(os.path.join(vdir, f))

# Code to draw rectangles
i = 0
for bounds in list2:
f = os.path.join(vdir, "cropped" + str(i) + ".jpg")
# draw.rectangle(bounds, width=2)
cropped = img.crop(bounds)
cropped.save(f, "PNG")
i = i+1
img.save('D:\Basket_binary_classification\modified_image\\basket (1).jpg', "PNG")

return draw, img, list1


def model(draw, img, list2):

classifier = Sequential()

# Multiple CNN layers
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten Layer to get a vector connect all the layers with the hidden layers
classifier.add(Flatten())
# Fully connected layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the model
classifier.compile(optimizer=optimizers.adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Part - 3 (Train the model)
# Code now to load our images in our model frst we do something to avoid overfitting
# train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.1, zoom_range = 0.2,
# horizontal_flip = True)
# test_datagen = ImageDataGenerator(rescale = 1./255)
# training_set = train_datagen.flow_from_directory('D:\\basketclass\dataset\\train', target_size = (64, 64),
# batch_size = 32, class_mode = 'binary')
# test_set = test_datagen.flow_from_directory('D:\\basketclass\dataset\\validation', target_size = (64, 64),
# batch_size = 32, class_mode = 'binary')

# Fitting data into model
# history=classifier.fit_generator(training_set,steps_per_epoch=28,epochs=19,validation_data=test_set,
# validation_steps=8, verbose=1)
# plt.figure(figsize=[8, 6])
# plt.plot(history.history['loss'],'r', linewidth=3.0, label="loss")
# plt.plot(history.history['val_loss'],'b', linewidth=3.0, label="val_loss")
# plt.xlabel('epoch', fontsize=16)
# plt.ylabel('loss', fontsize=16)
# plt.title('loss curve' , fontsize=16)
# plt.legend()
# plt.show()
classifier.load_weights("D:\Basket_binary_classification\weights\keras20.h5")

files = os.listdir('D:\Basket_binary_classification\crop_img')
list3 = []

for file in files:

test_image = image.load_img(os.path.join('D:\Basket_binary_classification\crop_img', file),
target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
data = test_image.astype("float32")
data /= 255
result = classifier.predict(data)
maximum = np.amin(result)
if maximum <= 0.02:
r = int(file[7:-4])
list3.append([int(list2[r][0]), int(list2[r][1]), int(list2[r][2]), int(list2[r][3])])
print("Predicted image:", file, "with accuracy: ", result)
for bounds in list3:
draw.rectangle(bounds, outline="#F8F806")
print("Image Modified!")
img.save('D:\Basket_binary_classification\modified_image\\basket (1).jpg', "PNG")
# d.click((newlist[r][0] + newlist[r][2]) / 2, (newlist[r][1] + newlist[r][3]) / 2)

# Main function


def main():
draw, img, list2 = image_pro()
model(draw, img, list2)


main()