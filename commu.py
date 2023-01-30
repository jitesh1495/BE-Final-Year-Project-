from PIL import Image
from numpy import asarray
import numpy
import serial
import cv2
import datetime

# sample.png is the name of the image
image = Image.open('/home/pi/Documents/Project/1.jpg')

# summarize some details about the image
print(image.format)
print(image.size)
print(image.mode)

np_img = numpy.array(image)
  
print(np_img.shape)

# load the image and convert into numpy array
numpydata = asarray(image)

      

# data
#print(numpydata)
print("size")
print(numpydata.size)
xbee = serial.Serial("/dev/ttyUSB0", 9600)
swt = True

while swt:
    incoming = str(numpydata)
    xbee.write(numpydata)
    
        
ct = datetime.datetime.now()    
f = open("demo.txt")
loc = "location1"
f.write(loc + " \n" + ct +" \n" + )

print("done sending")