from model import GMM_HMM
from datamaker import Datamaker
import numpy as np


class HMMRunner():

    def __init__(self):
        self.model = GMM_HMM()
        self.datamaker = Datamaker()

    def train(self, train_datadir, folders):
        train_dataset = self.datamaker.create_dataset(folders, train_datadir)
        models = {}

        for label in train_dataset.keys():
            model = GMM_HMM()
            train_data = train_dataset[label]

            train_data = np.vstack(train_data)
            model.train(train_data)
            models[label] = model
        return models

    def test(self, test_datadir, folders, models):
        test_dataset = self.datamaker.create_dataset(folders, test_datadir)

        acc = 0
        counter = 0

        for label in test_dataset.keys():
            test_data = test_dataset[label]
            score_list = {}
            counter += 1
            for model_label in models.keys():
                model = models[model_label]
                score = model.get_score(test_data[0])
                score_list[model_label] = score
            predict = max(score_list,key=score_list.get())
            if predict == label:
                acc += 1
        print("Accuracy {}".format(100 * acc/counter))