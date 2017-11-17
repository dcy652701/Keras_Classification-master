import cv2
import numpy as np
from keras.models import Sequential

from keras.models import load_model

img=[]
label=[]
label.append('bus')
label.append('dinosaur')
label.append('elephant')
label.append('flower')
label.append('horse')

model=Sequential()
model=load_model('model/demo.h5')
model.summary()
print "reading image...."
test_img=cv2.imread('dinosaur.jpg')
input_img=cv2.resize(test_img,(150,150))
print "load image done"
img.append(input_img)
np_img=np.array(img)
print "input img:",np_img.shape
print "\n"
#print "label:",label
res=model.predict(np_img)
print res
result=np.array(res)
for i in range(5):
    if result[0,i]>0:
        print "result is:",label[i],"\naccuracy is:",result[0,i],"\n"
