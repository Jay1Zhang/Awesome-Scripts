import os
import cv2
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

root_path = '/mnt/lustre/zhangjiayi/research/audio-visual/datasets/VGG-SS/'


def gen_vggss_csv():
    anno_path = root_path + 'vggss.json'
    with open(anno_path, 'r') as f:
        anno_list = json.load(f)
    
    # 删除vggsound中没有的数据
    meta = pd.read_csv('./temp/meta_clean.csv')
    meta = meta[meta['split'] == 'test']
    test_list = list(meta['filename'])

    video_list = []
    count = 0
    for anno in anno_list:
        filename = anno['file']
        category = anno['class']

        if filename not in test_list:
            continue
        
        video_list.append({
            'filename': filename,
            'category': category,
            'audio_path': f'audio_wav/{filename}.wav',
            'image_path': f'inter_frame/{filename}.png',
            'mask_path': f'masks/{filename}.png',
        })

        count += 1

    print('test_list:', len(test_list))
    print('anno_list:', len(anno_list))
    print('final masks num:', count)
    
    # read all data
    df = pd.DataFrame(video_list)
    df.to_csv(root_path + 'vggss.csv', sep=',', index=False)

    

def loas_mask(bbox_list):
    size = 224

    mask = np.zeros([size, size])
    for bbox in bbox_list:
        bbox =  list(map(lambda x: int(size * max(x,0)), bbox))
        temp = np.zeros([size, size])
        (xmin, ymin, xmax, ymax) = bbox[0], bbox[1], bbox[2], bbox[3]
        temp[ymin:ymax, xmin:xmax] = 1
        mask += temp

    mask[mask > 0] = 1
    return mask

def gen_masks():
    anno_path = root_path + 'vggss.json'
    with open(anno_path, 'r') as f:
        anno_list = json.load(f)

    meta = pd.read_csv('./temp/meta_clean.csv')
    meta = meta[meta['split'] == 'test']
    test_list = list(meta['filename'])
    
    count = 0
    for anno in tqdm(anno_list):
        filename = anno['file']
        category = anno['class']
        bbox_list = anno['bbox']

        if filename not in test_list:
            continue

        mask = loas_mask(bbox_list)
        mask_path = f'{root_path}/masks/{filename}.png'
        cv2.imwrite(mask_path, mask * 255)

        count += 1
    
    print('test_list:', len(test_list))
    print('anno_list:', len(anno_list))
    print('final masks num:', count)


def check():
    anno_path = root_path + 'vggss.json'
    with open(anno_path, 'r') as f:
        annos = json.load(f)

    anno_list = []
    for anno in annos:
        filename = anno['file']
        anno_list.append(filename)
    
    meta = pd.read_csv('./temp/meta_clean.csv')
    meta = meta[meta['split'] == 'test']
    meta['test'] = meta['filename'].map(lambda x: x in anno_list)
    # print(meta.head())
    # print(meta.info())
    
    print('anno_list:', len(anno_list))
    print('in test_list:', meta['test'].sum())
    assert meta['test'].sum() == len(anno_list)


if __name__ == "__main__":
    gen_vggss_csv()
    #gen_masks()
    #check()