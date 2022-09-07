import os
import io
import json
import pandas as pd
from tqdm import tqdm

import boto3
from petrel_client.client import Client


conf_path = '~/petreloss.conf'
S3Client = Client(conf_path)
S3Transfer = boto3.client('s3', endpoint_url='http://10.198.234.254:80')


def gen_category_map():
    onto_path = '/mnt/lustre/zhangjiayi/research/audio-visual/datasets/audioset/ontology.json'
    with open(onto_path, 'r') as f:
        onto = json.load(f)

    category_map = {}
    for category in onto:
        category_map[category['id']] = category['name']

    return category_map

def gen_meta_file(category_map):
    meta_path = '/mnt/lustre/zhangjiayi/research/audio-visual/datasets/audioset/unbalanced_train_segments.csv'
    
    output_path = '/mnt/lustre/zhangjiayi/research/audio-visual/datasets/audioset/audioset.csv'

    meta_raw = pd.read_csv(meta_path, encoding='utf-8', sep=', ')
    videos = []
    for index, item in tqdm(meta_raw.iterrows()):
        ytid = item['YTID']
        st = str(int(item['start_seconds'])*1000)
        ed = str(int(item['end_seconds'])*1000)
        filename = f'{ytid}_{st}_{ed}'

        audio_path = f'audio/{filename}.flac'
        video_path = f'video/{filename}.mp4'

        category_ids = item['positive_labels'][1:-1]  # 去除双引号 /m/02qldy,/m/02zsn,/m/05zppz,/m/09x0r
        category_ids = category_ids.split(',')

        categories = []
        for cid in category_ids:
            categories.append(category_map[cid])

        videos.append({
            'ytid': ytid,
            'filename': filename,
            'categories': categories,
            'audio_path': audio_path,
            'video_path': video_path,
        })
    
    meta = pd.DataFrame(videos)
    meta.to_csv(output_path, index=False)


def test_read():
    meta_path = '/mnt/lustre/zhangjiayi/research/audio-visual/datasets/audioset/audioset.csv'
    meta = pd.read_csv(meta_path)
    print(meta.info())
    print(meta.head())

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    #category_map = gen_category_map()
    #gen_meta_file(category_map)
    test_read()