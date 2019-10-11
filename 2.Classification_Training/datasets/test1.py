from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, glob


images_dir = "../../Project_Data/labeled_photos/"
list_images = [images_dir+f for f in os.listdir(images_dir)]

num_img=0

for folder in list_images:
    image_list=[]

    # subprocess.call(['bash ./convert.sh', folder], shell=True)
    image_list=glob.glob(folder+"/*.*")
    num_img+=len(image_list)
    





print(int(90*num_img/100))
print(num_img-int(90*num_img/100))
print(int(num_img))

print("A single integer between 0 and %d"%3 )
