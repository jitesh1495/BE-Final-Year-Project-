import cv2
import numpy as np
import PIL
from PIL import Image
from numpy import array
import re
import string
import serial
import datetime

# open or read the images
img1 = cv2.imread('Screenshot (121).png')
img2 = cv2.imread('Screenshot (122).png')

# resize the images to speed up processing
img1 = cv2.resize(img1,(640,480))
img2 = cv2.resize(img2,(640,480))


# display resized images
#cv2.imshow("Image 1", img1)
#cv2.imshow("Image 2", img2)

# convert imags to grayscale. This reduces matrices from 3 (R, G, B) to just 1
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# disply grayscale images
#cv2.imshow("Gray 1", gray1)
#cv2.imshow("Gray 2", gray2)

# blue the images to get rid of sharp edjes/outlines. This will improve the processing
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

# display blurred images
#cv2.imshow("Blur 1", gray1)
#cv2.imshow("Blur 2", gray2)

# obtain the difference between the two images & display the result
imgDelta = cv2.absdiff(gray1, gray2)
#cv2.imshow("Delta", imgDelta)

# coonvert the difference into binary & display the result
thresh = cv2.threshold(imgDelta, 25, 255, cv2.THRESH_BINARY)[1]
#cv2.imshow("Thresh", thresh)

# dilate the thresholded image to fill in holes & display the result
thresh = cv2.dilate(thresh, None, iterations=2)
#cv2.imshow("Dilate", thresh)

# find contours or continuous white blobs in the image
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# find the index of the largest contour
areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cnt=contours[max_index]

# draw a bounding box/rectangle around the largest contour
x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),2)
rectagle_crooped_img=img2[y:y+h,x:x+w]
#cv2.imshow('rectagle_crooped_img',rectagle_crooped_img)
cv2.imwrite('rectagle_crooped_img.jpeg',rectagle_crooped_img)

cv2.rectangle(thresh,(x,y),(x+w,y+h),(0,255,0),2)


crop_img=thresh[y:y+h,x:x+w]

#cv2.imshow("cropped",crop_img)
cv2.imwrite('crop_img.jpeg',crop_img)


image = cv2.imread('crop_img.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Set threshold level
threshold_level = 50

# Find coordinates of all pixels above threshold
coords = np.column_stack(np.where(gray > threshold_level))


# Create mask of all pixels lower than threshold level
mask = gray > threshold_level

# Color the pixels in the mask
image[mask] = (204, 119, 0)

cv2.imshow('image', image)

# display the original image for reference
#cv2.imshow("Show",img2)


# check for fire in RGB
rectagle_crooped_img1 = cv2.imread('rectagle_crooped_img.jpeg')
RGBfire = cv2.GaussianBlur(rectagle_crooped_img1, (21, 21), 0)
RGBfire1 = cv2.cvtColor(RGBfire, cv2.COLOR_BGR2RGB)
 
lower = [120,50,5]
upper = [250,120,10]
low = np.array(lower, dtype="uint8")
upp = np.array(upper, dtype="uint8")

maskRGB = cv2.inRange(RGBfire1, low, upp)
mask1 = cv2.bitwise_and(rectagle_crooped_img1,RGBfire1 , mask= maskRGB)
mask1[maskRGB>0] = (87, 20, 227)
cv2.imshow('test', rectagle_crooped_img1)
cv2.imshow('testRGBMask', mask1)
#cv2.imwrite('RGBfireImg.jpeg',mask1)

# check for fire in YCbCr
rectagle_crooped_img1 = cv2.imread('rectagle_crooped_img.jpeg')
YCCfire = cv2.GaussianBlur(rectagle_crooped_img1, (21, 21), 0)
YCCfire1 = cv2.cvtColor(YCCfire, cv2.COLOR_BGR2YCR_CB)
 
lower = [100,144,10]
upper = [250,210,60]
low = np.array(lower, dtype="uint8")
upp = np.array(upper, dtype="uint8")

maskYCC = cv2.inRange(YCCfire1, low, upp)
mask2 = cv2.bitwise_and(rectagle_crooped_img1,YCCfire1 , mask= maskYCC)
mask2[maskYCC>0] = (70, 40, 27)
#cv2.imshow('testYCC', rectagle_crooped_img1)
cv2.imshow('testYCCMask', mask2)
#cv2.imwrite('YCCfireImg.jpeg',mask2)

fire_signal = 0
#check for fire
if maskRGB.all() > 0 or maskYCC.all() > 0 :
    fire_signal = 1
    
print(fire_signal)
# check for smoke in RGB
rectagle_crooped_img = cv2.imread('rectagle_crooped_img.jpeg')
smoke = cv2.GaussianBlur(rectagle_crooped_img, (21, 21), 0)
smoke = cv2.cvtColor(smoke, cv2.COLOR_BGR2RGB)
 
PixelCoordinates = []
imgsmoke = PIL.Image.open('rectagle_crooped_img.jpeg')
width,height=imgsmoke.size
scount=0;
for x in range(0,width):
    for y in range(0,height):
        b0, g0, r0 = imgsmoke.getpixel((x,y))
        if r0 <= g0 <= b0 and 0 <= b0-r0 <= 15 :
            PixelCoordinates.append([x,y])
            scount= scount + 1;


smokepercentage = (scount/len(coords))*100
print(smokepercentage)
smoke_signal = 0
if smokepercentage > 1 :
    smoke_signal = 1

'''
# sample.png is the name of the image
image = Image.open('/home/pi/Documents/Project/1.jpg')

# summarize some details about the image
#print(image.format)
print(image.size)
#print(image.mode)

np_img = numpy.array(image)
  
#print(np_img.shape)

# load the image and convert into numpy array
numpydata = np(image)

      

# data
#print(numpydata)
print("size")
print(numpydata.size)
xbee = serial.Serial("/dev/ttyUSB0", 9600)
swt = True

while swt:
    incoming = str(numpydata)
    xbee.write(numpydata)
    
        
loc = "location1"        
ct = datetime.datetime.now()    
f = open(loc+ct.txt")

f.write(loc + " \n" + ct +" \n fire : " + fire_signal +" \n smoke : " + smoke_signal )


print("done sending")

'''
#wait for key press and then terminate all open windows
cv2.waitKey(0)
cv2.destroyAllWindows()
