import io
import sys
import mysql.connector
import binascii
from io import BytesIO
import serial
import codecs
import cv2
import numpy as np
import PIL
from PIL import Image
import base64
ser = serial.Serial('COM3', 9600,timeout=.5)
print ("done conn")
y= b''
y1=b''
y2=b''
y3=b""
yy=b''
z=0
swt=0
iname=""
idata=""


while swt<1:
   incoming = ser.readline().strip()
   
   if len(incoming)>0:
      y= y +incoming
      z1=len(y)-3
      if y[:-z1]==b"SIN" and y[z1:]==b'OIN' :
         y=y[3:]
         iname=y[0:-3]
         print(iname.decode())
      else :
         y=b""
         
   if len(incoming)>0:
      y1= y1 +incoming
      z2=len(y1)-4         
      if y1[:-z2]==b"SIMG" and y1[z2:]==b"OIMG":
         y1=y1[4:]
         idata=y1[0:-4]
         print(idata.decode())
         decodeit = open(iname.decode(), 'wb')
         decodeit.write(base64.b64decode((idata)))
         decodeit.close()
         
      else :
         y1=b""
         
   if len(incoming)>0:
      y2= y2 +incoming
      z1=len(y2)-3
      if y2[:-z1]==b"STN" and y2[z1:]==b"OTN":
         y2=y2[3:]
         tname=y2[0:-3]
         print(tname.decode())
      else :
         y2=b""
         
   if len(incoming)>0:
      y3= y3 +incoming
      z2=len(y3)-4
      if y3[:-z2]==b"STXT" and y3[z2:]==b"OTXT":
         y3=y3[4:]
         Tdata=y3[0:-4]
         print(Tdata.decode())
         ftxt = open(tname.decode(), 'w')
         ftxt.write(Tdata.decode())
         ftxt.close()
      else :
         y3=b""

print ("done")  
ser.close()
