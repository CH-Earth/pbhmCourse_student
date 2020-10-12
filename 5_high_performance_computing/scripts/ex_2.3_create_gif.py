# script to create a gif of individual images
import os
import imageio

# image folder
imDir = '../img/'

# find image files
files = []
for file in os.listdir(imDir):
    files.append(imDir+file)
    
# make sure the order is correct
files.sort()

# determine how many images to animate
iImg = 0
jImg = iImg + 4*365

# animate
images = []
for file in files[iImg:jImg]:
    images.append(imageio.imread(file))
imageio.mimsave('../animated_from_'+str(iImg)+'_to_'+str(jImg)+'.gif', images)