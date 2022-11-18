import cv2
import math
import os
import numpy as np

# Get user supplied values
cascPath = "haar.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
dir = "./images/"
dir_list = os.listdir(dir)

files = dir_list
all_images = []

for i in files:
  # Read the image
  image = cv2.imread(dir + i)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # Detect faces in the image
  faces = faceCascade.detectMultiScale(
      gray,
      scaleFactor=1.1,
      minNeighbors=5,
      minSize=(30, 30),
      #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
  )

  # Draw a rectangle around the faces
  for (x, y, w, h) in faces:
      cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

  all_images += [image]

  
# all_images = [all_images[2], all_images[1]]

w = 0
h = 0
for image in all_images:
  h1, w1 = image.shape[:2]
  print(w1)
  w += w1
  h = max(h, h1)

w//=3
vis = np.zeros((h*3, w,3), np.uint8)

wink = 0
hink = 0
i = 0
for image in all_images:
  row = 0
  row = math.floor(i/4) +1
  if i/4 == math.floor(i/4):
    wink = 0
  print(row)
  h2, w2 = image.shape[:2]
  vis[h2*(row-1):(h2*row), wink:(w2+wink), :3] = image
  wink += w2
  i+=1

cv2.imshow("Faces", vis)
cv2.waitKey(0)