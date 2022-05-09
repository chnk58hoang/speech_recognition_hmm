import os
import argparse
from glob import glob
import numpy as np
from scipy.io import wavfile
from hmmlearn import hmm
# from python_speech_features import mfcc
import librosa
import warnings

"""
class Datamaker use to get data and create dataset from files
"""


class Datamaker():
    def create_dataset(self, all_folders, data_dir):
        dataset = {}  # {label,mfcc features of label}
        for folder in all_folders:
            data_sub_dir = os.path.join(data_dir, folder)
            all_files = os.listdir(data_sub_dir)
            all_txts = [file for file in all_files if file.split('.')[1] == 'txt']

            all_data = self.get_data(all_txts, data_sub_dir)

            for data in all_data:
                wav_file = data['file'].replace('txt', 'wav')
                file_datas = data['data']

                for file_data in file_datas:
                    label, start, end = file_data
                    wav_path = os.path.join(data_sub_dir, wav_file)
                    audio, sr = librosa.load(wav_path, offset=start, duration=end - start)
                    mfcc = librosa.feature.mfcc(audio, sr)
                    mfcc = mfcc.transpose()
                    if label not in dataset.keys():
                        dataset[label] = []
                        dataset[label].append(mfcc)
                    else:
                        exits = dataset[label]
                        exits.append(mfcc)
                        dataset[label] = exits

        return dataset

    def get_data(self, all_txts, data_dir):
        """
        :param all_txts: list of all txt files
        :param data_dir: data directory
        :return: [{filename,(startime,endtime,label) of each word in audio file}]

        """
        all_file_data = []
        for file in all_txts:
            file_data = []
            with open(os.path.join(data_dir, file)) as lb:
                for line in lb.readlines():
                    line = [c for c in line.replace("\n", "").split()]
                    if len(line) == 3:
                        start, end, label = line
                        start = float(start)
                        end = float(end)
                        file_data.append((label, start, end))

            all_file_data.append({'file': file, 'data': file_data})
        return all_file_data
