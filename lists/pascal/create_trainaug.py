import os 
import pdb

raw_path = '/home/yz9244/PANet/data/pascal/VOCdevkit/VOC2012/ImageSets/Segmentation/trainaug.txt'
new_path = './trainaug.txt'
lines = open(raw_path).readlines()
new_f = open(new_path, 'w+')
print(len(lines))

for line in lines:   
    line = line.rstrip() 
    new_str = '/home/yz9244/PANet/data/pascal/VOCdevkit/VOC2012/JPEGImages/' + line + '.jpg' + ' /home/yz9244/PANet/data/pascal/VOCdevkit/VOC2012/SegmentationClassAug/' + line + '.png\n'
    new_f.write(new_str)
print('Finished.')