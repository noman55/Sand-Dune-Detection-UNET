import cv2
import numpy as np
from keras.models import load_model
import skimage.feature
import skimage.io
import math

IMAGE='2.png'
pass_density=1
vertical_mask_length=976
vertical_mask_height=976


image = skimage.io.imread(IMAGE, as_grey=True)
image_for_output = skimage.io.imread(IMAGE, as_grey=True)

image = skimage.feature.local_binary_pattern(image=image, P=10, R=20)

store=np.array([],dtype=int)
coordinates=np.array([],dtype=int)
batch_predict_size=10

def return_batch(index):
    global array_index
    global store
    global batch_predict_size
    if((batch_predict_size*(index+1))>len(array_index)):
        a=len(array_index)%batch_predict_size
        arr=np.empty((a,976,976,1))
        b=int(len(array_index)/batch_predict_size)
        for i in range(a):
            arr[i,:,:,:]=store[(b*batch_predict_size)+i,:,:,:]
        print("gg")
        return arr
    else:
        arr=np.empty((batch_predict_size,976,976,1))
        for i in range(10):
            arr[i,:,:,:]=store[(index*batch_predict_size)+i,:,:,:]
        print("hh")
        return arr

        
    
    
    

def store_coordinates(y_start,y_end,x_start,x_end):
    global image
    global coordinates
    global store
    global count
    coordinates=np.append(coordinates,[y_start, x_start, y_end, x_end])
    image_data=image[y_start: y_end, x_start : x_end] 
    store=np.append(store,image_data)
    print("stored!")
    
#HORIZONTAL
number_of_whole_passes_possible_x=int(image.shape[1]/vertical_mask_length)     #we will leave the last pass as its traversed in the 2nd last 
density_passes_made_x=pass_density*number_of_whole_passes_possible_x
x_spacing=int(vertical_mask_length/pass_density)

#VERTICAL
number_of_whole_passes_possible_y=int(image.shape[0]/vertical_mask_height)       #we will leave the last pass as its traversed in the 2nd last 
density_passes_made_y=pass_density*number_of_whole_passes_possible_y
y_spacing=int(vertical_mask_height/pass_density)


for i in range(density_passes_made_x):
    x_start=i*x_spacing
    for j in range(density_passes_made_y):
        y_start=j*y_spacing
        #image_sliced=image[y_start: (y_start+vertical_mask_height), x_start : (x_start + vertical_mask_length)]      #OPPOSITE FORMAT!!    (HEIGHT!!!=ROWS=X)
        store_coordinates(y_start,y_start+vertical_mask_height,x_start,x_start + vertical_mask_length)

print(store_coordinates)
store=np.reshape(store,(-1,976,976,1))

#often the culprit
store=store/255.0
array_index=np.arange(store.shape[0])

model = load_model('my_model_lbp_10_20_976px.h5')
coordinates=np.reshape(coordinates,(-1,4))


indx=math.ceil(len(array_index)/batch_predict_size)
print(indx)

for k in range(indx):
    result=model.predict(return_batch(k))

    for i in range(result.shape[0]):
        for x in range(976):
            for y in range(976):
                if(result[i,x,y,0]>0.5):
                    result[i,x,y,0]=255
                else:
                    result[i,x,y,0]=0

    for j in range(result.shape[0]):
        y_start=coordinates[(k*batch_predict_size)+j,0]
        y_end= coordinates[(k*batch_predict_size)+j,2]
        x_start=coordinates[(k*batch_predict_size)+j,1]
        x_end=coordinates[(k*batch_predict_size)+j,3]
        image_for_output[y_start:y_end,x_start:x_end]=result[j,:,:,0]

   
   
'''
img=np.empty((976,976)) 
for i in range(store.shape[0]):
    img=result[i,:,:,0]
    cv2_imshow(img)
'''
'''   
cv2.imshow('',image_for_output)
cv2.waitKey()
cv2.destroyAllWindows()   
'''
cv2.imwrite("random.png",image_for_output)

    
    
    
    
    
    
    
    