import os
import cv2
import time
import librosa
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms

#! modify, 表示可能需要修改的地方
from utils import *
from opts import get_opt
from model import Location, make_model_parallel
from dataloader import VGG3SDataset


def load_raw_audio(path, nsec, fps, nframe):
    samples, samplerate = librosa.load(path, sr=16000, mono=True)
    # repeat if audio is too short
    if len(samples) < samplerate * nsec:    # 取前nec秒，不足的要补齐
        n = int(samplerate * nsec / len(samples)) + 1
        samples = np.tile(samples, n)       # 沿时间维度扩大n倍，相当于把n段重复音频拼接在一起
    samples = samples[:samplerate*nsec]
    # stft
    spec = librosa.stft(samples, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='reflect')
    # repeat
    spec = np.expand_dims(spec, axis=0)   # 扩展维度 (1, H, W)
    spec = np.expand_dims(spec, axis=0).repeat(nframe, axis=0)   # (5, 1, H, W)
    spec = spec.astype(np.float32)
    return torch.tensor(spec)


def load_log_mel(path, nsec, fps, nframe):
    spec = pickle.load(open(path, 'rb')).requires_grad_(False)  # (5,1,96,64)

    h, w = spec.shape[-2], spec.shape[-1]
    # torchvggish 预处理得到的spec是按秒划分的，需要分段重复fps次，以对齐视频帧
    spec = [torch.repeat_interleave(spec[i], repeats=fps, dim=0).unsqueeze(1) for i in range(nsec)]   # [fps,1,h,w]
    spec = torch.stack(spec).reshape(nframe, 1, h, w)
    return spec


def load_data(video_path, audio_path, use_raw_audio):
    size = (224, 224)
    # 1. open video 
    cap = cv2.VideoCapture(video_path) 
    
    # nsec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)) 
    # fps = int(nframe / nsec)    # cap.get(cv2.CAP_PROP_FPS) 
    # nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # note: 由于可能有fps=7.5的情况，为保证fps为整数（需要repeat）因此仅保留前nsec*fps帧
    nsec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    nframe = int(nsec * fps)    
    #assert int(nsec * fps) == nframe
    print(f'real nsec: {cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)}, fps: {cap.get(cv2.CAP_PROP_FPS)}, nframe: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}')
    print(f'adopted nsec: {nsec}, fps: {fps}, nframe: {nframe}')
    
    # 2. get frames and images
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]) 

    frames = []
    images = []
    frame_no = 0
    flag, frame = cap.read()
    while(flag and frame_no < nframe):
        frame = cv2.resize(frame, size, cv2.INTER_LINEAR)
        image = img_transform(frame)
        frames.append(frame)
        images.append(image)

        flag, frame = cap.read()
        frame_no += 1

    frames = np.stack(frames)   # (nframe, 224, 224, 3)
    images = torch.stack(images)   # (nframe, 3, 224, 224)

    # 3. get spectrograms
    spec = load_raw_audio(audio_path, nsec, fps, nframe) if use_raw_audio else load_log_mel(audio_path, nsec, fps, nframe)
    
    assert images.shape[0] == spec.shape[0]

    return (images, spec, frames, nsec, fps, nframe)


def inference(model, img, spec):
    print('Inference')
    model.eval()

    #! modify
    #upsampler = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
    pred_list = []
    with torch.no_grad():
        N = img.shape[0]

        avc, cls_a, cls_v, fine_a, fine_v, cam_a, cam_v, aln_a, aln_v = model(spec.float(), img.float())
        cls_a = torch.nn.functional.sigmoid(cls_a)      # (N, 23)
        ## dist 
        similarity = torch.einsum('bij,bik->bjk', fine_a, fine_v)
        similarity = torch.nn.functional.relu(similarity)
        similarity = similarity.view(*similarity.shape[:2], 14, 14)
        similarity = torch.nn.functional.interpolate(similarity, (224, 224), 
                                                    mode='bilinear')        # (N, 23, 224, 224)
        ## cam
        cam_a = torch.nn.functional.interpolate(cam_a, (224, 224), 
                                                    mode='bilinear')
        cam_v = torch.nn.functional.interpolate(cam_v, (224, 224), 
                                                    mode='bilinear')        # (N, 23, 224, 224)
        
        for i in range(N):
            threshold = 0.3
            pred = localize(img[i], similarity[i], cls_a[i], threshold)
            pred[pred > threshold]  = 1
            pred[pred < 1] = 0
            pred_list.append(pred)
        
    return pred_list


def pipeline():
    #! modify
    opt = get_opt()
    config_device(opt)

    data_path = '/mnt/lustre/zhangjiayi/research/datasets/VGG3S_Dataset/'
    output_path = '/mnt/lustre/zhangjiayi/research/demo/'

    name = '9zwmS4D3xQU'    # ss_test___with_audio
    category = 'baby_laughter'
    split = 'test'
    datasets = 'ss'  # multi_sources
    use_raw_audio = True
    add_audio = True
    device = opt.device

    # 0. set input video path
    if datasets == 'ss':
        video_path = f'{data_path}/single_source/trimed_video/{split}/{category}/{name}.mp4'
        audio_path = f'{data_path}/single_source/audio_wav/{split}/{category}/{name}.wav'
        log_mel_path = f'{data_path}/single_source/audio_log_mel/{split}/{category}/{name}.pkl'
        demo_path = f'{output_path}/ss_{split}_{category}_{name}.mp4'
        demo_with_audio_path = f'{output_path}/ss_{split}_{category}_{name}_with_audio.mp4'
    else:
        video_path = f'{data_path}/multi_sources/clipped_videos//{name}.mp4'
        audio_path = f'{data_path}/multi_sources/multi_audio_wav/{split}/{name}.wav'
        log_mel_path = f'{data_path}/multi_sources/multi_audio_log_mel/{split}/{name}.pkl'
        demo_path = f'{output_path}/ms_{split}_{name}.mp4'
        demo_with_audio_path = f'{output_path}/ms_{split}_{name}_with_audio.mp4'
    

    # 1. load data
    print('load data...')
    audio_path = audio_path if use_raw_audio else log_mel_path
    (img, spec, frames, nsec, fps, nframe) = load_data(video_path, audio_path, use_raw_audio)
    img, spec = img.to(device), spec.to(device)

    # 2. load model
    #! modify
    print('load model...')
    model = Location() 
    model = resume_model(opt, model)
    model = make_model_parallel(model, opt.distributed, opt.device)

    # 3. inference and get predmap
    print('predict graymap...')
    pred_list = inference(model, img, spec)

    # 4. write demo video
    print('write video...')
    size = (224, 224)
    cmap = [146, 122, 255]
    videowriter = cv2.VideoWriter(demo_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(nframe):
        frame = frames[i]
        pred = pred_list[i] # (224, 224)
        pred = np.expand_dims(pred, axis=-1).repeat(3, axis=-1)     # (224, 224, 3)
        pred = (pred * np.array(cmap)).astype(np.uint8)
        
        add_image = cv2.addWeighted(frame, 1, pred, 0.6, 0)     # 图像1;图像1透明度(权重);图像2;图像2透明度(权重);叠加后图像亮度
        videowriter.write(add_image)

    videowriter.release()

    # 5. combine video and audio
    print('add audio...')
    if add_audio:
        cmd = f'ffmpeg -i {audio_path} -i {demo_path} {demo_with_audio_path}'
        os.system(cmd)
        #print(cmd)
        print('video with audio: ', demo_with_audio_path)

    print('raw video: ', demo_path)
    print('done!')


if __name__ == '__main__':
    pipeline()
