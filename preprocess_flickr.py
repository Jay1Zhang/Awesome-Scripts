import os
import cv2
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

root_path = '/mnt/lustre/zhangjiayi/research/audio-visual/datasets/Flickr-SoundNet/'

def gen_flickr_csv():
    anno_path = root_path + 'Annotations/'
    data_list = []
    for filename in os.listdir(anno_path):
        name = filename[:-4]
        data_list.append(name)
    
    # read all data
    df_all = pd.DataFrame(data_list, columns=['name'])
    df_all['split'] = 'train'
    # read test data
    df_test = pd.read_csv(root_path + 'flickr_test.csv', header=None)
    df_test['name'] = df_test[0].apply(lambda x: str(x))    # int -> str
    # map test in all data
    #df_all['test'] = df_all['name'].map(lambda x: x in df_test['name'])    # bug here
    test_list = list(df_test['name'])
    df_all['test'] = df_all['name'].map(lambda x: x in test_list)
    df_all.loc[df_all['test'], 'split'] = 'test'

    # gen path
    df_all['audio_path'] = df_all['name'].map(lambda x: f'Data/{x}.wav')
    df_all['image_path'] = df_all['name'].map(lambda x: f'Data/{x}.jpg')
    df_all['mask_path'] = df_all['name'].map(lambda x: f'Masks/{x}.png')

    df_all[['name', 'split', 'audio_path', 'image_path', 'mask_path']].to_csv(root_path + 'flickr.csv', sep=',', index=False)

    

def _loas_mask(path, name):
    # Learning to Localize Sound Source in Visual Scenes原文中抽取gt_map的方式
    gt = ET.parse(f'{path}{name}.xml').getroot()
    gt_map = np.zeros([224,224])
    bboxs = []
    for child in gt: 
        for childs in child:
            bbox = []
            if childs.tag == 'bbox':
                for index,ch in enumerate(childs):
                    if index == 0:
                        continue
                    bbox.append(int(224 * int(ch.text)/256))
            bboxs.append(bbox)

    for item_ in bboxs:
        temp = np.zeros([224,224])
        (xmin,ymin,xmax,ymax) = item_[0],item_[1],item_[2],item_[3]
        temp[item_[1]:item_[3],item_[0]:item_[2]] = 1
        gt_map += temp

    gt_map /= 2
    gt_map[gt_map>1] = 1

    return gt_map

def gen_masks():
    meta_path = root_path + 'flickr.csv'
    anno_path = root_path + 'Annotations/'
    mask_path = root_path + 'Masks/'

    with open(meta_path, 'r') as f:
        content = f.read().strip().replace(',train', '').replace(',test', '')
        file_list = content.split('\n')[1:]
        print(len(file_list))

    for filename in file_list:
        mask = _loas_mask(anno_path, filename)
        cv2.imwrite(f'{mask_path}{filename}.png', mask * 255)


if __name__ == "__main__":
    gen_flickr_csv()
    #gen_masks()