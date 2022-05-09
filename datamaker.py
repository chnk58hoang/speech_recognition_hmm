import os
import argparse
from glob import glob
import numpy as np
from scipy.io import wavfile
from hmmlearn import hmm
from python_speech_features import mfcc
import warnings


class Datamaker():

    def create_dataset(self, all_folders,data_dir):
        dataset = list()
        for folder in all_folders:
            data_sub_dir = os.path.join(data_dir, folder)
            all_files = os.listdir(data_sub_dir)
            all_files = [file for file in all_files if file.split('.')[1] == 'wav']

            for file in all_files:
                sampling_freq, audio = wavfile.read(file)
                label = self.get_label(file)
                dataset.append({"label": label, "mfcc": mfcc(audio, sampling_freq, nfft=13)})

        return dataset

    def get_label(self, file):
        file = file.split('.')[0]
        if file.find('c') != -1:
            file.replace('c', '')
            return file

        else:
            for i in range(len(file) - 1, -1, -1):
                if (file[i] != '0' and file[i - 1] == '0') or i == 0:
                    file = file[i:]
                    return file
