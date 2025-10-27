"""
python preprocess/extract_audio.py -input_file_list /storage/slurm/schnackl/fakesync/data/mavosdd/mavos_indomain.csv -target_fold /storage/slurm/schnackl/fakesync/data/mavosdd/preprocessed
"""
import os
import numpy as np
import argparse
from tqdm import tqdm
import re
import subprocess

parser = argparse.ArgumentParser(description='Easy video feature extractor')
parser.add_argument("-input_file_list", type=str, default='sample_video_extract_list.csv', help="Should be a csv file of a single columns, each row is the input video path.")
parser.add_argument("-target_fold", type=str, default='./sample_audio/', help="The place to store the video frames.")
args = parser.parse_args()

#input_filelist = np.loadtxt(args.input_file_list, delimiter=',', dtype=str)
# load column named video_name from csv
import pandas as pd
df = pd.read_csv(args.input_file_list)
input_filelist = df['video_name'].to_numpy()

# input_filelist = input_filelist[:10]

if os.path.exists(args.target_fold) == False:
    os.makedirs(args.target_fold)

# first resample audio
for i in tqdm(range(input_filelist.shape[0]), desc='Resampling audio to 16kHz mono'):
    input_f = input_filelist[i]
    ext_len = len(input_f.split('/')[-1].split('.')[-1])
    # video_id = input_f.split('/')[-1][:-ext_len-1]
    video_id = "-".join(input_f.split('/')[-5:])[:-ext_len-1]
    # video_id = re.escape(video_id)
    output_f_1 = args.target_fold + '/' + video_id + '_intermediate.wav'
    os.system("ffmpeg -i '{:s}' -vn -loglevel error -ar 16000 '{:s}'".format(input_f, output_f_1)) # save an intermediate file
    # subprocess.run(['ffmpeg','-i', input_f, '-vn','-loglevel','error','-ar','16000', output_f_1], check=True)

# then extract the first channel
for i in tqdm(range(input_filelist.shape[0]), desc='Extracting first audio channel'):
    input_f = input_filelist[i]
    ext_len = len(input_f.split('/')[-1].split('.')[-1])
    # video_id = input_f.split('/')[-1][:-ext_len-1]
    video_id = "-".join(input_f.split('/')[-5:])[:-ext_len-1]
    # video_id = re.escape(video_id)
    output_f_1 = args.target_fold + '/' + video_id + '_intermediate.wav'
    output_f_2 = args.target_fold + '/' + video_id + '.wav'
    os.system("sox '{:s}' '{:s}' remix 1".format(output_f_1, output_f_2))
    # subprocess.run(['sox', output_f_1, output_f_2, 'remix', '1'])
    # remove the intermediate file
    try: 
        os.remove(output_f_1)
    except OSError:
        print("Error while deleting file : ", output_f_1)
