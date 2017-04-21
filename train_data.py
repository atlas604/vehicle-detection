import os
import glob

basedir = 'vehicles/'
image_types = os.listdir(basedir)

cars = []
for imtype in image_types:
        cars.extend(glob.glob(basedir+imtype+'/*'))

print ('Number of Vehicle Images found:', len(cars))
with open ("cars.txt", 'w') as f:
    for fn in cars:
        f.write(fn+'\n')

basedir = 'non-vehicles/'
image_types = os.listdir(basedir)

notcars = []
for imtype in image_types:
        notcars.extend(glob.glob(basedir+imtype+'/*'))

print ('Number of Non-Vehicle Images found:', len(notcars))
with open ("notcars.txt", 'w') as f:
    for fn in notcars:
        f.write(fn+'\n')
