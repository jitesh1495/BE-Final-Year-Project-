import cv2
import numpy as np
import PIL
from PIL import Image
from numpy import array
import re
import string

# open or read the images
img1 = cv2.imread('Screenshot (119).png')
img2 = cv2.imread('Screenshot (120).png')

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
cv2.imshow("Show",img2)



# check for fire in RGB
rectagle_crooped_img1 = cv2.imread('rectagle_crooped_img.jpeg')
RGBfire = cv2.GaussianBlur(rectagle_crooped_img1, (21, 21), 0)
RGBfire1 = cv2.cvtColor(RGBfire, cv2.COLOR_BGR2RGB)
 
lower = [10,10,10]
upper = [250,250,250]
low = np.array(lower, dtype="uint8")
upp = np.array(upper, dtype="uint8")

maskRGB = cv2.inRange(RGBfire1, low, upp)
mask1 = cv2.bitwise_and(rectagle_crooped_img1,RGBfire1 , mask= maskRGB)
#mask1[maskRGB>0] = (227, 20, 62)
cv2.imshow('testRGB', rectagle_crooped_img1)
cv2.imshow('testRGBMask', mask1)
cv2.imwrite('RGBfireImg.jpeg',mask1)

# check for fire in YCbCr

rectagle_crooped_img2 = cv2.imread('rectagle_crooped_img.jpeg')
YCCtest = cv2.GaussianBlur(rectagle_crooped_img2, (21, 21), 0)
YCCtest1 = cv2.cvtColor(YCCtest, cv2.COLOR_BGR2YCR_CB)
 
Ylow = [0.06,0.08,0.08]
Yupp = [0.90,0.98,0.98]
lower1 = np.array(Ylow)
upper1 = np.array(Yupp)
maskYCC = cv2.inRange(YCCtest1, lower1, upper1)
mask2 = cv2.bitwise_and(rectagle_crooped_img2, YCCtest, mask= maskYCC)
mask2[maskYCC > 0] = (254,254,6)
cv2.imshow('testYCC', rectagle_crooped_img2)
cv2.imshow('testYCCMask', mask2)
cv2.imwrite('YccfireImg.jpeg',mask2)


'''
# check for smoke in RGB
rectagle_crooped_img = cv2.imread('rectagle_crooped_img.jpeg')
smoke = cv2.GaussianBlur(rectagle_crooped_img, (21, 21), 0)
smoke = cv2.cvtColor(smoke, cv2.COLOR_BGR2RGB)
 
PixelCoordinates = []
imgsmoke = PIL.Image.open('rectagle_crooped_img.jpeg')
width,height=imgsmoke.size
for x in range(0,width):
    for y in range(0,height):
        r0, g0, b0 = imgsmoke.getpixel((x,y))
        if r0 <= g0 <= b0 and 0 <= b0-r0 <= 15 :
            PixelCoordinates.append([x,y])

PixelCoordinates=np.array(PixelCoordinates)
smoke1=cv2.imread('rectagle_cropped_img.jpeg')
smokeS = cv2.GaussianBlur(smoke1, (21, 21), 0)
smokeS = cv2.cvtColor(smokeS, cv2.COLOR_BGR2RGB)
mask3 = cv2.bitwise_and(smokeS, smokeS, mask= mask)
mask3[PixelCoordinates] = (50,55,57)
cv2.imshow('testSmoke', smoke1)
cv2.imshow('testRGBMaskSmoke', mask3)
cv2.imwrite('SmokeImg.jpeg',smoke3)

#finding similar coordinates 
#maskSmoke = cv2.inRange(smoke, lower2, upper2)
p10 = cv2.imread('RGBfireImg.jpeg')
p20 = cv2.imread('YCCfireImg.jpeg')
#p30 = cv2.imread('SmokeImg.jpeg')
FireMaskR = (227, 20, 62)
FireMaskY = (254,254,6)
SmokeMaskR = (50,55,57)

indices0 = np.where(p10 == FireMaskR)
indices1 = np.where(p20 == FireMaskY)
#indices2 = np.where(p30 == SmokeMaskR)
indicesU= np.union1d(indices0,indices1)
#indices=np.union1d(indicesU,indices2)

coordinates = zip(indicesU)
unique_coordinates = list(set(list(coordinates)))
listCoords=coords.tolist()

#def intersection(lst1,lst2):
    #return list(set(lst1) and set(lst2))
#f1=intersection(unique_coordinates,listCoords)
#matchpercentage = (len(f1)/len(listCoords))*100
#print("mactch percentage: ", matchpercentage)
'''

'''
for i in coords[:6800]:
    j=str(i)
    n = np.char.replace(j,'[','(')
    p = np.char.replace(n,']',')')
    t=str(p)
    mid=(len(t)+1)/2
    m=int(mid-1)
    new = int(t[1:m], base=10)
    few = int(t[int(mid+1):len(t)-1],base=10)
    Cropped_image = Image.open("YCbCr.jpeg")
    yccb_pixel_value = Cropped_image.getpixel((new,few))
    #Cropped_image.getdata()
    #print(Cropped_image.getdata())
    #print(yccb_pixel_value)
    #print(new,few)

    #colour=yccb_pixel_value.split(',')
    c0=yccb_pixel_value[0]
    c1=yccb_pixel_value[1]
    c2=yccb_pixel_value[2]
'''
    









#wait for key press and then terminate all open windows
cv2.waitKey(0)
cv2.destroyAllWindows()
