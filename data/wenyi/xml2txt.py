import os
import xml.etree.ElementTree as ET
import random
from os import getcwd

# 在train.txt和val.txt文件中生成各自数据集的图片绝对路径
# 再新建一个labels文件夹,其中以每张图片id为文件名生成该张图片的标注数据txt文档
# 每张txt的数据格式
# cls_id ymin xmin ymax max
# cls_id ymin xmin ymax max
# cls_id ymin xmin ymax max

sets = ['train', 'val']

classes = ["WhitehairedBanshee", "UndeadSkeleton", "WhitehairedMonster", "SlurryMonster", "MiniZalu", "Dopelliwin",
           "ShieldAxe", "SkeletonKnight","Zalu","Cyclone","SlurryBeggar","Gerozaru","Catalog",
           "InfectedMonst","Gold","StormRider","Close","Door",]
classes_num = 18
# 当前路径
data_path = getcwd()
def convert_annotation(image_id):
    in_file = open(image_id.replace('JPGImages','Annotations').replace('jpg','xml'),'r')
    out_file = open(image_id.replace('JPGImages','labels').replace('jpg','txt'),'w')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('ymin').text), float(xmlbox.find('xmin').text), float(xmlbox.find('ymax').text),
             float(xmlbox.find('xmax').text))
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in b]) + '\n')


trainval_percent = 1
train_percent = 0.9
xmlfilepath = 'Annotations'
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrain = open('train.txt', 'w')
fval = open('val.txt', 'w')
for i in list:
    name = os.path.join(getcwd(),'JPGImages',total_xml[i][:-4]+'.jpg')
    if i in train:
        ftrain.write(name+'\n')
    else:
        fval.write(name+'\n')
ftrain.close()
fval.close()

for image_set in sets:
    # 如果labels文件夹不存在则创建
    if not os.path.exists(data_path+'\labels\\'):
        os.makedirs(data_path+'\labels\\')

    image_ids = open(data_path+'\%s.txt' % (image_set)).read().strip().split()
    for image_id in image_ids:
        convert_annotation(image_id)
