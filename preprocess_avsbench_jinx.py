import os, glob, sys
import pandas as pd
import numpy as np
import random, pickle
from tqdm import tqdm
import imageio
from moviepy.editor import VideoFileClip
import cv2
import shutil
import torch
sys.path.append("../../VGG3S/torchvggish/")
from torchvggish import vggish_input
import json

import pdb

VGG3SDataset = ['baby_laughter', 'female_singing', 'male_speech',
'dog_barking', 'cat_meowing', 'lions_roaring', 'horse_clip-clop', 'coyote_howling', 'mynah_bird_singing',
'playing_acoustic_guitar', 'playing_tabla', 'playing_violin', 'playing_piano', 'playing_ukulele', 'playing_glockenspiel',
'helicopter', 'race_car', 'driving_buses', 'ambulance_siren',
'typing_on_computer_keyboard', 'chainsawing_trees', 'cap_gun_shooting', 'lawn_mowing'] # 23 categories


def add_category_for_each_csv(val_test_ratio=0.15):
    clean_anno_path = "./clean_anno"
    if not os.path.exists(clean_anno_path):
        os.mkdir(clean_anno_path)
    dir_and_file_name_list = os.listdir('./')
    csv_list = []
    for item in dir_and_file_name_list:
        if item.endswith('.csv'):
            csv_list.append(item)
    print(csv_list)
    for file_name in csv_list:
        df_one_csv_data = pd.read_csv(file_name, sep=',')
        df_one_csv_data['category'] = file_name.split('.')[0]
        # df_one_csv_data['split'] = 'train'
        lenth = len(df_one_csv_data)
        val_num  = round(val_test_ratio * lenth)
        test_num = val_num
        ori_flag = ['train'] * lenth
        for i in range(val_num):
            ori_flag[i] = "val"
        for i in range(1, test_num + 1):
            ori_flag[-i] = "test"
        # pdb.set_trace()
        random.shuffle(ori_flag)
        df_one_csv_data['split'] = ori_flag
        df_one_csv_data.to_csv(os.path.join(clean_anno_path, file_name), index=None)
        # pdb.set_trace()

def merge_csv_file(path="./clean_anno"):
    all_csv_files = glob.glob(os.path.join(path, "*.csv"))
    df_from_each_file = (pd.read_csv(f, sep=',') for f in all_csv_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=True)
    df_merged['start'] = df_merged['start'].values.astype(int)
    # df_merged.to_csv(os.path.join(path, "VGG3S_anno.csv"), index=None)
    df_merged.to_csv(os.path.join("../VGG3S_anno.csv"), index=None)
    from collections import Counter
    pdb.set_trace()

def trim_for_each_video(video_path, trimed_video_base_path, df_one_video):
    video_name, start_time, category, split = df_one_video[0], df_one_video[1], df_one_video[2], df_one_video[3]
    ori_video_path = os.path.join(video_path, category, video_name + ".mp4")
    trimed_video_path = os.path.join(trimed_video_base_path, split, category, video_name + ".mp4")
    if not os.path.exists(os.path.join(trimed_video_base_path, split, category)):
        os.makedirs(os.path.join(trimed_video_base_path, split, category))
    # pdb.set_trace()

    t_start = start_time
    t_end = t_start + 5 if (t_start <= 5) else 10
    t_dur = t_end - t_start
    # print("*******************trim the video to [%.1f-%.1f]" % (t_start, t_end))
    command2 = 'ffmpeg '
    command2 += '-ss '
    command2 += str(t_start) + ' '
    command2 += '-i '
    command2 += ori_video_path + ' '
    command2 += '-t '
    command2 += str(t_dur) + ' '
    command2 += '-vcodec libx264 '
    command2 += '-acodec aac -strict -2 '
    command2 += trimed_video_path + ' '
    command2 += '-y '  # overwrite without asking
    command2 += '-loglevel -8 '  # print no log
    #print(command2)
    os.system(command2)
    # print ("!!!!!!!!!! finish the video as: " + trimed_video_path)
    # pdb.set_trace()

def trim_videos():
    video_path = "../video"
    trimed_video_path = "../trimed_video"
    anno_path = "../VGG3S_anno.csv"
    df_anno = pd.read_csv(anno_path, sep=',')
    wrong_video_list = []
    count = 0
    for item in VGG3SDataset:
    # for item in ['playing_tabla', 'playing_ukulele', 'chainsawing_trees']:
    # for item in ['cat_meowing']:
        print(f"processing for [{item}]...")
        df_one_class = df_anno[df_anno['category'] == item]
        # pdb.set_trace()
        for i in tqdm(range(len(df_one_class))):
            try:
                trim_for_each_video(video_path, trimed_video_path, df_one_class.iloc[i])
                count += 1
            except Exception as e:
                print(f"Error {e} -> {df_one_class[i]}")
                wrong_video_list.append(df_one_class[i])
    pdb.set_trace()



def video_frame_sample(frame_interval, video_length, sample_num):
    num = []
    for l in range(video_length):
        for i in range(sample_num):
            num.append(int(l * frame_interval + (i * 1.0 / sample_num) * frame_interval))
    return num


def extract_frame_for_each_video(trimed_video_base_path, df_one_video):
    video_name, start_time, category, split = df_one_video[0], df_one_video[1], df_one_video[2], df_one_video[3]
    trimed_video_path = os.path.join(trimed_video_base_path, split, category, video_name + ".mp4")
    extract_frames_base_path = "../trimed_frames_png"

    # t = 5 # length of video
    t = 5 if (start_time <= 5) else (10 - start_time)

    sample_num = 16 # frame number for each second
    c = 0

    vid = imageio.get_reader(trimed_video_path, 'ffmpeg')
    # vid_len = len(vid)
    # frame_interval = int(vid_len / t)
    frame_interval = int(round(vid.get_meta_data()['fps']))
    # fps = int(round(vid.get_meta_data()['fps']))

    frame_num = video_frame_sample(frame_interval, t, sample_num)
    imgs = []
    for i, im in enumerate(vid):
        x_im = cv2.resize(im, (224, 224))
        imgs.append(x_im)
    vid.close()
    # print('processing video_name [%s], len_imgs [%d], len_frame [%d]'%(os.path.join(split, category, video_name), len(imgs), len(frame_num)))
    # pdb.set_trace()

    frame_save_path = os.path.join(extract_frames_base_path, split, category, video_name)
    if not os.path.exists(frame_save_path):
        os.makedirs(frame_save_path)

    extract_frame = []
    for n in frame_num:
        if n >= len(imgs):
            n = len(imgs) - 1
        # print(n)
        extract_frame.append(imgs[n])
    # pdb.set_trace()

    count = 0
    for k, frame in enumerate(extract_frame):
        if k % sample_num == 15:
            count += 1
            cv2.imwrite(os.path.join(frame_save_path, video_name + '_' + str(count) + '.png'), cv2.cvtColor(extract_frame[k], cv2.COLOR_RGB2BGR))
            # pdb.set_trace()

    # pdb.set_trace()


def extract_frames():
    trimed_video_path = "../trimed_video"
    anno_path = "../VGG3S_anno.csv"
    df_anno = pd.read_csv(anno_path, sep=',')
    wrong_video_list = []
    count = 0
    for item in VGG3SDataset:
    # for item in VGG3SDataset[13:]:
    # for item in ['playing_piano']:
        print(f"processing for [{item}]...")
        df_one_class = df_anno[df_anno['category'] == item]
        for i in tqdm(range(len(df_one_class))):
            try:
                extract_frame_for_each_video(trimed_video_path, df_one_class.iloc[i])
                count += 1
            except Exception as e:
                print(f"Error {e}")
                print(df_one_class[i])
                wrong_video_list.append(df_one_class[i])
    pdb.set_trace()


def count_num4each_class(check_path='../trimed_frames/'):
    total_num = 0
    for split in ['train', 'val', 'test']:
    #for i in range(1):
        print(f"=================== {split} ==============")
        for item in VGG3SDataset:
            # lenth_one_class = len(os.listdir(os.path.join('../trimed_video/' + split, item)))
            lenth_one_class = len(os.listdir(os.path.join(check_path, split, item)))
            #lenth_one_class = len(os.listdir(os.path.join(check_path, item)))
            total_num += lenth_one_class
            print(f"#{item}: {lenth_one_class}")
    print(f'total_num: {total_num}')
    print('=' * 20)
    pdb.set_trace()


def copy_video_to_frames_dir_for_seg_anno():
    trimed_video_path = '../trimed_video'
    trimed_frames_for_seg_path = '../trimed_frames_for_seg_anno'
    # for the 'train' set, remove the 2rd-5th image, copy the video to current images directory
    split = 'train'
    for category in tqdm(os.listdir(os.path.join(trimed_frames_for_seg_path, split))):
        for video_name in tqdm(os.listdir(os.path.join(trimed_frames_for_seg_path, split, category))):
            try:
                for idx in ['2', '3', '4', '5']:
                    os.remove(os.path.join(trimed_frames_for_seg_path, split, category, video_name, "%s_%s.jpg"%(video_name, idx)))
            except FileNotFoundError:
                pass
            shutil.copy(os.path.join(trimed_video_path, split, category, "%s.mp4"%(video_name)), os.path.join(trimed_frames_for_seg_path, split, category, video_name, "%s.mp4"%video_name))
            # pdb.set_trace()
    # for the 'val' and 'test' sets, copy the video to current images directory
    for split in ['val', 'test']:
        for category in tqdm(os.listdir(os.path.join(trimed_frames_for_seg_path, split))):
            for video_name in tqdm(os.listdir(os.path.join(trimed_frames_for_seg_path, split, category))):
                shutil.copy(os.path.join(trimed_video_path, split, category, "%s.mp4"%(video_name)), os.path.join(trimed_frames_for_seg_path, split, category, video_name, "%s.mp4"%video_name))
        # pdb.set_trace()

def count_num4trimed_frames_for_seg_anno():
    # 等同于命令： ls ./train -lR| grep "^-" | wc -l
    split = 'train'
    trimed_frames_for_seg_path = '../trimed_frames_for_seg_anno'
    dict = {'train': 0, 'val': 0, 'test': 0}
    for split in ['train', 'val', 'test']:
        for category in os.listdir(os.path.join(trimed_frames_for_seg_path, split)):
            for video_name in os.listdir(os.path.join(trimed_frames_for_seg_path, split, category)):
                file_list = os.listdir(os.path.join(trimed_frames_for_seg_path, split, category, video_name))
                if split == 'train' and len(file_list) != 2:
                    print('Wrong files for ', os.path.join(trimed_frames_for_seg_path, split, category, video_name))
                if split != 'train' and len(file_list) != 6:
                    print('Wrong files for ', os.path.join(trimed_frames_for_seg_path, split, category, video_name))
                dict[split] += len(file_list)
    ''' 这几个数据是从第6秒开始，只抽取了4张图片
    Wrong files for  ../trimed_frames_for_seg_anno/test/playing_tabla/bC8vaaJF-kk
    Wrong files for  ../trimed_frames_for_seg_anno/test/ambulance_siren/BtrmVCFYLYU
    Wrong files for  ../trimed_frames_for_seg_anno/test/ambulance_siren/NCnxDcaltzo
    Wrong files for  ../trimed_frames_for_seg_anno/test/lions_roaring/DPlSp0M9Mgs
    Wrong files for  ../trimed_frames_for_seg_anno/test/cat_meowing/JTbensAt_SE
    Wrong files for  ../trimed_frames_for_seg_anno/test/cap_gun_shooting/2eEPiCh9bZo
    train set里面也有8个只6秒的'''

    # 最终处理后： {'train': 6920, 'val': 4440, 'test': 4440} 处理验证正常
    pdb.set_trace()

def deal_with_extract_frames_having_only_4files():
    to_check_dir = "../trimed_frames_png"
    anno_path = "../VGG3S_anno.csv"
    df_anno = pd.read_csv(anno_path, sep=',')
    wrong_video_list = []
    df_4files = df_anno[df_anno['start'] == 6]
    print('%d videos only have 4 frames.'%len(df_4files))
    for i in range(len(df_4files)):
        df_one_video = df_4files.iloc[i]
        video_name, start_time, category, split = df_one_video[0], df_one_video[1], df_one_video[2], df_one_video[3]
        src_dir = os.path.join(to_check_dir, split, category, video_name, video_name+'_4.png')
        des_dir = os.path.join(to_check_dir, split, category, video_name, video_name+'_5.png')
        shutil.copyfile(src_dir, des_dir)
        pdb.set_trace()


def split_audio(trimed_video_base_path, wav_save_base_path, df_one_video):
    """extract the .wav file from one video"""
    video_name, start_time, category, split = df_one_video[0], df_one_video[1], df_one_video[2], df_one_video[3]
    trimed_video_path = os.path.join(trimed_video_base_path, split, category, video_name + ".mp4")
    wav_save_path = os.path.join(wav_save_base_path, split, category, video_name + ".wav")
    if not os.path.exists(os.path.join(wav_save_base_path, split, category)):
        os.makedirs(os.path.join(wav_save_base_path, split, category))
    # pdb.set_trace()
    video = VideoFileClip(trimed_video_path)
    audio = video.audio
    audio.write_audiofile(wav_save_path, fps=16000)

def extract_audio_wav(wav_save_base_path="../audio_wav"):
    """extract the .wav files for videos in SSS DATASET"""
    trimed_video_path = "../trimed_video"
    anno_path = "../VGG3S_anno.csv"
    df_anno = pd.read_csv(anno_path, sep=',')
    wrong_video_list = []
    count = 0
    for item in VGG3SDataset:
    # for item in VGG3SDataset[13:]:
    # for item in ['playing_piano']:
        print(f"processing for [{item}]...")
        df_one_class = df_anno[df_anno['category'] == item]
        for i in tqdm(range(len(df_one_class))):
            try:
                split_audio(trimed_video_path, wav_save_base_path, df_one_class.iloc[i])
                count += 1
            except Exception as e:
                print(f"Error {e}")
                print(df_one_class[i])
                wrong_video_list.append(df_one_class[i])
    print('wrong_list: ', wrong_video_list)
    print('#wrong_list: ', len(wrong_video_list))


def extract_one_log_mel(wav_base_path, lm_save_base_path, df_one_video):
    """extract the .wav file from one video and save to .pkl"""
    video_name, start_time, category, split = df_one_video[0], df_one_video[1], df_one_video[2], df_one_video[3]
    wav_path = os.path.join(wav_base_path, split, category, video_name + ".wav")
    lm_save_path = os.path.join(lm_save_base_path, split, category, video_name + ".pkl")
    if not os.path.exists(os.path.join(lm_save_base_path, split, category)):
        os.makedirs(os.path.join(lm_save_base_path, split, category))
    # pdb.set_trace()

    log_mel_tensor = vggish_input.wavfile_to_examples(wav_path)
    count_4sec = 0
    wrong_list = []
    if log_mel_tensor.shape[0] != 5:
        wrong_list.append(df_one_video)
        print('start time: ', start_time)
        print('video_name: ', video_name)
        print('lm.shape: ', log_mel_tensor.shape)
        N_SECONDS, CHANNEL, N_BINS, N_BANDS = log_mel_tensor.shape
        new_lm_tensor = torch.zeros(5, CHANNEL, N_BINS, N_BANDS)
        new_lm_tensor[:N_SECONDS] = log_mel_tensor
        new_lm_tensor[N_SECONDS:] = log_mel_tensor[-1].repeat(5-N_SECONDS, 1, 1, 1)
        log_mel_tensor = new_lm_tensor

    with open(lm_save_path, "wb") as fw:
        pickle.dump(log_mel_tensor, fw)
    # with open(lm_save_path, 'rb') as fr:
    #     lm_data = pickle.load(fr)
    # pdb.set_trace()
    return wrong_list

def extract_audio_log_mel(lm_save_base_path="../audio_log_mel"):
    """extract and save the log_mel map for each .wav"""
    wav_path = "../audio_wav"
    anno_path = "../VGG3S_anno.csv"
    df_anno = pd.read_csv(anno_path, sep=',')
    wrong_video_list = []
    video_no5s_list = []
    count = 0
    for item in VGG3SDataset:
        print(f"processing for [{item}]...")
        df_one_class = df_anno[df_anno['category'] == item]
        for i in tqdm(range(len(df_one_class))):
            try:
                wrong_item = extract_one_log_mel(wav_path, lm_save_base_path, df_one_class.iloc[i])
                video_no5s_list.extend(wrong_item)
                count += 1
            except Exception as e:
                print(f"Error {e}")
                print(df_one_class[i])
                wrong_video_list.append(df_one_class[i])
    print('wrong_list: ', wrong_video_list)
    print('#wrong_list: ', len(wrong_video_list))
    print('video_no5s_list: ', video_no5s_list)
    print('#video_no5s_list: ', len(video_no5s_list))



def generate_one_mask_from_json(mask_save_path, json_file, video_name, idx=1):
    """generate mask given one json file"""
# def generate_one_mask_from_json(mask_save_path='../temp_masks', json_file='../trimed_frames_anno_json/train/male_speech/6VfPJDACvn8/6VfPJDACvn8_1.jpg.json', video_name="_Gd04M1s1wg", idx=1):
    # if not os.path.exists(mask_save_path):
    #     os.mkdir(mask_save_path)
    with open(json_file) as fr:
        anno_data = json.load(fr)

    # pdb.set_trace()
    num_objs = len(anno_data["step_1"]["result"])
    mask_list = []
    for i in range(num_objs):
        point_num = len(anno_data["step_1"]["result"][i]["pointList"])
        point_list = np.zeros((point_num, 2))
        for kp_idx in range(point_num):
            point_list[kp_idx, 0] = anno_data["step_1"]["result"][i]["pointList"][kp_idx]['x']
            point_list[kp_idx, 1] = anno_data["step_1"]["result"][i]["pointList"][kp_idx]['y']
        point_array = np.asarray(point_list, np.int32)
        mask_list.append(point_array)
    mask = np.zeros((224, 224))
    # pdb.set_trace()
    cv2.fillPoly(mask, mask_list, (255, 255, 255))
    cv2.imwrite(os.path.join(mask_save_path, "%s_%d.png"%(video_name, idx)), mask)
    # print("saving to ", os.path.join(mask_save_path, "%s_%d.png"%(video_name, idx)))


def process_mask_for_one_video(mask_save_base_path, df_one_video):
    """ generate mask according to the json file"""
    frames_anno_json_path = "../trimed_frames_anno_json"
    video_name, start_time, category, split = df_one_video[0], df_one_video[1], df_one_video[2], df_one_video[3]
    mask_save_path = os.path.join(mask_save_base_path, split, category, video_name)
    if not os.path.exists(mask_save_path):
        os.makedirs(mask_save_path)
    json_base_file = os.path.join(frames_anno_json_path, split, category, video_name)
    if split == 'train':
        json_file = os.path.join(json_base_file, "%s_1.jpg.json"%video_name)
        generate_one_mask_from_json(mask_save_path, json_file, video_name, idx=1)
    else:
        for idx in range(1, 6):
            json_file = os.path.join(json_base_file, "%s_%d.jpg.json"%(video_name, idx))
            generate_one_mask_from_json(mask_save_path, json_file, video_name, idx=idx)


def generate_masks(mask_save_base_path="../trimed_frames_masks"):
    if not os.path.exists(mask_save_base_path):
        os.mkdir(mask_save_base_path)
    anno_path = "../VGG3S_anno.csv"
    df_anno = pd.read_csv(anno_path, sep=',')
    wrong_video_list = []
    count = 0
    for item in VGG3SDataset:
        print(f"processing for [{item}]...")
        df_one_class = df_anno[df_anno['category'] == item]
        for i in tqdm(range(len(df_one_class))):
            try:
                process_mask_for_one_video(mask_save_base_path, df_one_class.iloc[i])
                count += 1
            except Exception as e:
                print(f"Error {e}")
                print(df_one_class[i])
                wrong_video_list.append(df_one_class[i])
    print('wrong_list: ', wrong_video_list)
    print('#wrong_list: ', len(wrong_video_list))




if __name__ == "__main__":
    # add_category_for_each_csv()
    # merge_csv_file()
    # trim_videos()
    # extract_frames()
    deal_with_extract_frames_having_only_4files()

    # count_num4each_class('../trimed_video/')
    # count_num4each_class('../trimed_frames/')
    # count_num4each_class('../trimed_frames_for_seg_anno/')
    # count_num4each_class('./video/')
    # count_num4each_class('../audio_wav/')
    # count_num4each_class('../audio_log_mel/')

    # copy_video_to_frames_dir_for_seg_anno()
    # count_num4trimed_frames_for_seg_anno()
    # extract_audio_wav()
    # extract_audio_log_mel()

    # generate_one_mask_from_json()
    # generate_masks()



# 以下27个video都是音频提取log_mel的时候，不足5秒, 之后标注结果返回后，需要重新修改下相应的mask标注
# video_no5s_list:  [name           2GuH1UJxQOk
# start                    5
# category    female_singing
# split                 test
# Name: 3797, dtype: object, name        84vGUGey4pM
# start                 5
# category    male_speech
# split              test
# Name: 2150, dtype: object, name        1QxDi-op6Qg
# start                 5
# category    cat_meowing
# split              test
# Name: 3558, dtype: object, name        6-tqMSNhbrs
# start                 5
# category    cat_meowing
# split             train
# Name: 3597, dtype: object, name        FtNV_Gq62l8
# start                 5
# category    cat_meowing
# split             train
# Name: 3678, dtype: object, name        JTbensAt_SE
# start                 6
# category    cat_meowing
# split              test
# Name: 3688, dtype: object, name          d-adbXZYHxI
# start                   6
# category    lions_roaring
# split               train
# Name: 1689, dtype: object, name          DPlSp0M9Mgs
# start                   6
# category    lions_roaring
# split                test
# Name: 1691, dtype: object, name          DpOPNpRDZMk
# start                   6
# category    lions_roaring
# split               train
# Name: 1692, dtype: object, name          sPm0vT9fyoY
# start                   6
# category    lions_roaring
# split               train
# Name: 1810, dtype: object, name            3sf-Fa7cZfA
# start                     5
# category    horse_clip-clop
# split                 train
# Name: 4510, dtype: object, name                    0ioRPfQzAW0
# start                             6
# category    playing_acoustic_guitar
# split                         train
# Name: 3294, dtype: object, name          bC8vaaJF-kk
# start                   6
# category    playing_tabla
# split                test
# Name: 2539, dtype: object, name            _dRqfb20DdU
# start                     6
# category    playing_ukulele
# split                 train
# Name: 2245, dtype: object, name            b2Vm8epWPXc
# start                     5
# category    playing_ukulele
# split                   val
# Name: 2287, dtype: object, name            vhUM-UOKpSk
# start                     4
# category    playing_ukulele
# split                   val
# Name: 2406, dtype: object, name        4UdR7R1RHh0
# start                 6
# category       race_car
# split             train
# Name: 4782, dtype: object, name        8M1AO0uEYck
# start                 6
# category       race_car
# split             train
# Name: 4810, dtype: object, name        IaCgOqKsqW8
# start                 5
# category       race_car
# split             train
# Name: 4891, dtype: object, name          _sZzRgGlHvg
# start                   5
# category    driving_buses
# split               train
# Name: 2724, dtype: object, name          L6UfR2Wlnuw
# start                   5
# category    driving_buses
# split                test
# Name: 2810, dtype: object, name          Yk08kMxQ7jU
# start                   5
# category    driving_buses
# split               train
# Name: 2858, dtype: object, name            36AsCpqS3z0
# start                     6
# category    ambulance_siren
# split                 train
# Name: 4093, dtype: object, name            BtrmVCFYLYU
# start                     6
# category    ambulance_siren
# split                  test
# Name: 4109, dtype: object, name            NCnxDcaltzo
# start                     6
# category    ambulance_siren
# split                  test
# Name: 4174, dtype: object, name                        TDfjgkUGglE
# start                                 5
# category    typing_on_computer_keyboard
# split                              test
# Name: 745, dtype: object, name             2eEPiCh9bZo
# start                      6
# category    cap_gun_shooting
# split                   test
# Name: 2878, dtype: object]
# video_no5s_list:  27