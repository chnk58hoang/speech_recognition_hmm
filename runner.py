from model import GMM_HMM
from datamaker import Datamaker
import numpy as np
import os
import pickle

"""
class for training and validating models
"""

class HMMRunner():

    def __init__(self):
        self.model = GMM_HMM()
        self.datamaker = Datamaker()

    def train(self, train_datadir, folders):
        print('Training...')
        train_dataset = self.datamaker.create_dataset(folders, train_datadir)

        # fix the wrong labels
        for label in train_dataset.keys():
            if label == 'a':
                label = 'A'
            elif label == 'b':
                label = 'B'
            elif label == 'si':
                label = 'sil'
            elif label == 'ba':
                label = 'ban'

            model = GMM_HMM() # one model is corresponding to one label
            train_data = train_dataset[label]

            train_data = np.vstack(train_data)
            model.train(train_data)
            model_file = label + '.pkl'
            #Save model
            with open(model_file, "wb") as file:
                pickle.dump(model, file)

    def test(self, test_datadir, folders):
        print('Testing...')
        test_dataset = self.datamaker.create_dataset(folders, test_datadir)

        all_files = os.listdir('/content/drive/MyDrive/speech_recognition_hmm')
        model_files = [file for file in all_files if len(file.split('.')) == 2 and file.split('.')[1] == 'pkl']

        acc = 0
        counter = 0

        for label in test_dataset.keys():
            test_datas = test_dataset[label]
            score_list = {}
            counter += 1
            for model_file in model_files:
                model_label = model_file.split('.')[0]
                with open(model_file, "rb") as file:
                    model = pickle.load(file)

                    score = model.get_score(test_datas[0])
                    score_list[model_label] = score
            predict = max(score_list, key=score_list.get)
            if predict == label:
                acc += 1

        print("Accuracy %.3f" % (100 * acc / counter), '%')


if __name__ == '__main__':
    runner = HMMRunner()
    data_dir = '/content/drive/MyDrive/09'
    folders = os.listdir(data_dir)
    runner.train(data_dir, folders[:-1])
    runner.test(data_dir, folders[:-1])
