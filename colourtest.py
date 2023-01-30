import cv2
import numpy as np
import PIL
from PIL import Image
from numpy import array
import re
import string


# check for fire in RGB
rectagle_crooped_img1 = cv2.imread('2-detected.jpg')
RGBfire = cv2.GaussianBlur(rectagle_crooped_img1, (21, 21), 0)
RGBfire1 = cv2.cvtColor(RGBfire, cv2.COLOR_BGR2RGB)

lower = [190,120,0]
upper = [255,240,50]
low = np.array(lower, dtype="uint8")
upp = np.array(upper, dtype="uint8")

maskRGB = cv2.inRange(RGBfire1, low, upp)
mask1 = cv2.bitwise_and(rectagle_crooped_img1,RGBfire1 , mask= maskRGB)
mask1[maskRGB>0] = (87, 20, 227)
#cv2.imshow('testRGB', rectagle_crooped_img1)
cv2.imshow('testRGBMask', mask1)
cv2.imwrite('RGBfireImg.jpeg',mask1)

# check for fire in YCbCr
#rectagle_crooped_img1 = cv2.imread('1-detected.jpeg')
YCCfire = cv2.GaussianBlur(rectagle_crooped_img1, (21, 21), 0)
YCCfire1 = cv2.cvtColor(YCCfire, cv2.COLOR_BGR2YCR_CB)
#LH[100,16h,10][250,240,60l]l[0,150,0]h[255,255,120]nwl[140,150,0]nwh[255,255,53]
lower = [0,0,0]
upper = [230,255,255]
low = np.array(lower, dtype="uint8")
upp = np.array(upper, dtype="uint8")

maskYCC = cv2.inRange(YCCfire1, low, upp)
mask2 = cv2.bitwise_and(rectagle_crooped_img1,YCCfire1 , mask= maskYCC)
mask2[maskYCC>0] = (70, 40, 27)
cv2.imshow('testYCC', rectagle_crooped_img1)
cv2.imshow('testYCCMask', mask2)
cv2.imwrite('YCCfireImg.jpeg',mask2)




# check for smoke in RGB
rectagle_crooped_img = cv2.imread('1.jpeg')
smoke = cv2.GaussianBlur(rectagle_crooped_img, (21, 21), 0)
smoke = cv2.cvtColor(smoke, cv2.COLOR_BGR2RGB)
 
PixelCoordinates = []
imgsmoke = PIL.Image.open('1.jpeg')
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


#wait for key press and then terminate all open windows
cv2.waitKey(0)
cv2.destroyAllWindows()
