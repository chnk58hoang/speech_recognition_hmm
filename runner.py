from model import GMM_HMM
from datamaker import Datamaker


class HMMRunner():

    def __init__(self):
        self.model = GMM_HMM()
        self.datamaker = Datamaker()

    def train(self, train_datadir, folders):
        train_dataset = self.datamaker.create_dataset(folders, train_datadir)
        training_files = list()
        for sample in train_dataset:
            training_files.append({'label': sample["label"], 'feature': sample["mfcc"]})

        for training_file in training_files:
            X = training_file['feature']
            model = GMM_HMM()
            model.train(X)
            training_file['model'] = model

        return training_files

    def test(self, test_datadir, folders, traing_files):
        test_dataset = self.datamaker.create_dataset(folders, test_datadir)
        counter = 0
        acc = 0
        for sample in test_dataset:
            counter += 1
            max_score = None
            true_label = sample['label']
            for item in traing_files:
                model = item['model']
                score = model.get_score(sample['mfcc'])
                if max_score is None or score > max_score:
                    max_score = score
                    predicted = item['label']

            if predicted == true_label:
                acc += 1
            print('Predict:{0},Label: {1}'.format(predicted, true_label))
        acc = 100 * (acc / counter)
        print('Accuracy:{}'.format(acc))


if __name__ == '__main__':
    runner = HMMRunner()