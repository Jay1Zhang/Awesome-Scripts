import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from petrel_client.client import Client


conf_path = '~/petreloss.conf'
S3Client = Client(conf_path)


def get_data_set(url):
    files = S3Client.get_file_iterator(url)
    data_set = set()
    
    for p, k in tqdm(files):
        s3url = k['Key']                    # 'vggsound/audio_wav/---g-f_I2yQ_000001.wav'
        filename = s3url.split('/')[-1]     # '---g-f_I2yQ_000001.wav'
        filename = filename.split('.')[0]   # '---g-f_I2yQ_000001'
        
        data_set.add(filename)
       
    return data_set


def clean_meta(input_path, output_path):
    # Read meta file
    with open(input_path, 'rb') as f:
        contents = f.read().decode('utf-8')
        
    # Process meta and generate df
    print('is processing meta data...')
    videos = contents.strip().split('\n')[1:]    # drop title
    video_list = []
    for v in tqdm(videos):
        v = v.replace(', ', '!@#$').split(',')   # replace ', ' to special character

        vid = v[0]
        split = v[1]
        category = v[2].replace('!@#$', ', ')    # inreplace
        
        st_sec = v[3].split('/')[-1].split('_')[-2] # '1000'
        st_sec = str(int(st_sec) // 1000).zfill(6)  # -> '000001'
        filename = f'{vid}_{st_sec}'        # '---g-f_I2yQ_000001'
        
        video_path = v[3].replace('data/', '')
        audio_path = f'audio_wav/{filename}.wav'
        inter_frame_path = f'inter_frame/{filename}.png'

        video_list.append({
            'vid': vid,
            'split': split,
            'category': category,
            'filename': filename,
            'video_path': video_path,
            'audio_path': audio_path,
            'inter_frame_path': inter_frame_path
        })
    
    df = pd.DataFrame(video_list)

    # Drop Bad Data
    print('is dropping bad data...')
    # toooooo slowly!
    # for index, audio_path, inter_frame_path in zip(df.index, df['audio_path'], df['inter_frame_path']):
    #     s3url_audio = f's3://mmg_data_audio/{audio_path}'
    #     s3url_inter_frame = f's3://mmg_data_audio/{inter_frame_path}'

    #     if not (S3Client.contains(s3url_audio) and S3Client.contains(s3url_audio)):
    #         df.drop(labels=index, inplace=True)
    
    # df.reset_index(drop=True, inplace=True)
    audio_set = get_data_set('cluster1:s3://mmg_data_audio/vggsound/audio_wav/')
    frame_set = get_data_set('cluster1:s3://mmg_data_audio/vggsound/inter_frame/')
    print('audio_set:', len(audio_set))
    print('frame_set:', len(frame_set))
    drop_list = []
    for index, filename in tqdm(zip(df.index, df['filename'])):     # 179846it
        if not (filename in audio_set and filename in frame_set):
            drop_list.append(index)

    df.drop(labels=drop_list, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Sample ValSet, No Need
    # train_df = df[df['split'] == 'train']
    # val_df = train_df.sample(frac=0.1, replace=False, random_state=1)
    # df.loc[df.index.isin(val_df.index), 'split'] = 'val'
    
    # Write clean meta file
    df.to_csv(output_path, index=False)
    

def random_samples(meta_path, sample_num='10k'):
    meta = pd.read_csv(meta_path)
    meta = meta[meta['split'] == 'train']
    print(meta.head())
    # random sampling
    np.random.seed(10)
    if sample_num == '10k':
        sample_meta = meta.sample(10000)    # 默认replace=False，即不放回抽样
    elif sample_num == '144k':
        sample_meta = meta.sample(144000)

    sample_meta.reset_index(drop=True, inplace=True)
    sample_meta.to_csv(f'./temp/vggsound_{sample_num}.csv', index=False)



if __name__ == '__main__':
    
    #clean_meta('./temp/meta.csv', './temp/meta_clean.csv')
    random_samples('./temp/meta_clean.csv', '144k')