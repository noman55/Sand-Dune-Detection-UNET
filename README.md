# Sand-Dune-Detection-UNET
This is used to detect Sand Dunes on Martian Surface Here I have used Image Processing techique called LBP(local binary 
pattern), the reason being it is illumination invarient it is (LBP) is texture based technique where ever similar texture maches it 
will consider it as same texture so, basically shadow effect is gone here


data contain two sets one is input which is lbp images of (976px by 976px) and other is masks which contains the masks of lbp images 
of same size 

there are two folders one is train on which model is trained and other is valid on which classification or validation accuracy is checked
UNET.py file contains the UNET model 

results were applied on 2.png image and output is observed in random.png image
