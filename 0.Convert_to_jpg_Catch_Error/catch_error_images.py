# remove damaged images in subfolders in image_dir

import imghdr,subprocess
import glob
import os, re
from PIL import Image

images_dir = "../Project_Data/labeled_photos/"
list_images = [images_dir+f for f in os.listdir(images_dir)]

for folder in list_images:
    image_list=[]
    image_list=glob.glob(folder+"/*.*")
    for img in image_list:

        if str(imghdr.what(img))!="jpeg" :
            if str(imghdr.what(img)) != "None":
                print(imghdr.what(img))
                
                print(img)
                os.remove(img)
     




