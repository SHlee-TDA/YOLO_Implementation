import json
import os
import torch
import random
import xml.etree.ElemntTree as ET
import torchvision.transforms.functional as FT
from label_transform import *

# Device 설정
def prepare_device(n_gpu_use):
    '''
    GPU 사용 설정을 하는 함수입니다. 

        Args :
            n_gpu_use (int) : 사용할 GPU의 수
        Return :
            device (str) : 사용중인 장비의 종류 (cuda or cpu)
            list_ids (list) : 사용중인 gpu의 id 리스트 (gpu는 앞에서부터 순서대로 사용합니다.)
    '''
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == .0:
        print("경고 : 현재 기기에 사용 가능한 GPU가 없습니다.\n"
              "학습은 CPU를 이용해 진행됩니다.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu :
        print(f"경고 : 설정한 GPU의 수 {n_gpu_use}개가 기기에서 사용 가능한 GPU의 수 {n_gpu}개보다 많습니다.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')    
    list_ids = list(range(n_gpu_use))
    return device, list_ids

# Label map
voc_labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
              'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person',
              'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

labels = ['background'] + voc_labels
label_map = LabelTransform(labels).index_to_label
index_map = LabelTransform(labels).label_to_index

def parse_annotation(annotation_path):
    '''
    Annotation 데이터를 파싱합니다.
        
        Args :
            annotation_path (str) : Annotation (xml) 데이터가 들어있는 디렉토리
        Return :
            dictionary (dic) : Python dictionary 형태로 파싱된 데이터
    '''
    tree = ET.parse(annotation_path)    # annotation_path에 있는 데이터를 파싱합니다.
    root = tree.getroot()               # 

    boxes = list()
    labels = list()
    difficulties = list()

    for object in root.iter('object'):
        
        difficult = int(object.find('difficult').text == '1')

        label = object.find('name').text.lower().strip()
        if label not in label_map:
            continue

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {'boxes' : boxes, 'labels' : labels, 'difficulties' : difficulties}

def create_data_list(voc12_path, output_folder):
    '''
    이미지 데이터들의 리스트를 만듭니다.

        Args :
            voc12_path (str) : VOC 2012 데이터가 포함된 path
            output_folder (str) : 출력된 json 데이터가 저장될 폴더
        Return :
    '''
    voc12_path = os.path.abspath(voc12_path)

    train_images = list()
    train_objects = list()
    n_objects = 0

    # Training data
    
    # Find IDs of images in training data
    with open(os.path.join(voc12_path, 'ImageSets/Main/trainval.txt')) as f:
        ids = f.read().splitlines()

        
