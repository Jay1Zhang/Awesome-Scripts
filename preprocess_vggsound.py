import os
import subprocess
import multiprocessing

import boto3
from petrel_client.client import Client


conf_path = '~/petreloss.conf'
S3Client = Client(conf_path)
S3Transfer = boto3.client('s3', endpoint_url='http://10.198.234.254:80')

# # Extra
# with open('temp/remaining.txt', 'rb') as f:
#     existing_list = f.read().decode('utf-8').strip().split('\n')
# existing_set = set(existing_list)
# # Extra end

def get_video_list(url):
    files = S3Client.get_file_iterator(url)
    video_list = []
    sep = '_'
    for p, k in files:
        s3url = k['Key']                    # 'vggsound/video/---g-f_I2yQ_1000_11000.mp4'
        filename = s3url.split('/')[-1]     # '---g-f_I2yQ_1000_11000.mp4'
        rindex = filename.rfind(sep, 0, filename.rfind(sep))# 11
        vid = filename[:rindex]             # '---g-f_I2yQ'
        st_sec = filename.split(sep)[-2]                    # '1000'
        st_sec = str(int(st_sec) // 1000).zfill(6)          # '000001'
        filename = f'{vid}_{st_sec}'        # '---g-f_I2yQ_000001'
        
        video_list.append({
            'vid': vid, 
            's3url': s3url,
            'filename': filename
        })
       
    return video_list


def pipeline(video):
    # video = {'vid': '---g-f_I2yQ',
    #          'filename': '---g-f_I2yQ_000001',
    #          's3url': 'vggsound/video/---g-f_I2yQ_1000_11000.mp4'}

    try:
        vid = video['vid']
        filename = video['filename']

        # # Extra
        # if filename in existing_set:
        #     print(filename, 'existed, skip.')
        #     return 
        # # End

        s3url_video = video['s3url']
        s3url_audio = f'vggsound/audio_wav/{filename}.wav'
        s3url_inter_frame = f'vggsound/inter_frame/{filename}.png'

        video_path = f'temp/{filename}.mp4'
        audio_path = f'temp/{filename}.wav'
        inter_frame_path = f'temp/{filename}.png'

        print('is processing video:', filename, flush=True)
        # Download
        S3Transfer.download_file('mmg_data_audio', s3url_video, video_path) # bucketName, objectName, fileName

        # Clip the audio
        subprocess.check_call([
            'ffmpeg',
            '-y',
            '-i', video_path,
            '-acodec', 'pcm_s16le',
            '-ac','1',
            '-ar','16000',
            '-loglevel', 'quiet',
            audio_path
        ])

        # Clip video
        subprocess.check_call([
            'ffmpeg',
            '-y',
            '-ss', '5',
            '-i', video_path,
            '-vframes', '1',
            '-s','224x224',
            '-loglevel', 'quiet',
            inter_frame_path
        ])

        # Upload
        S3Transfer.upload_file(audio_path, 'mmg_data_audio', s3url_audio)
        S3Transfer.upload_file(inter_frame_path, 'mmg_data_audio', s3url_inter_frame)

        # Delete
        os.remove(video_path)
        os.remove(audio_path)
        os.remove(inter_frame_path)

        with open('temp/remaining.txt', 'a') as f:
            f.write(f'{filename}\n')
    
    except Exception as e:
        print('Unknown Exception Occurred when processing vid:', vid)
        print(e)
        #os.remove(video_path)

        with open('temp/missing.txt', 'a') as f:
            f.write(f'{filename}\n')


def preproc_video(num_worker):
    video_list = get_video_list('cluster1:s3://mmg_data_audio/vggsound/video/')
    pool = multiprocessing.Pool(num_worker)
    pool.map(pipeline, video_list)
    print('Processed Finish, Total Videos:', len(video_list))


if __name__ == '__main__':
    # spring.submit arun -n 1 --cpus-per-task 32 'python preprocess_vggsound.py' 
    preproc_video(num_worker=32)
    