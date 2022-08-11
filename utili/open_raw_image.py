import rawpy
import cv2 as cv

path = 'image.raw'
raw = rawpy.imread(path)
rgb = raw.postprocess()
cv.imwrite('ammm.jpg', rgb)