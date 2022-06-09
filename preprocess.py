from torchvision.transforms import transforms
from pathlib import Path
import matplotlib.pyplot as plt
import h5py
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np


def _loader(data, label):
    data_set = TensorDataset(torch.tensor(data.astype(np.float), dtype=torch.float),
                             torch.tensor(label.astype(np.float), dtype=torch.float))
    data_loader = DataLoader(data_set, shuffle=False, batch_size=1)

    return data_loader


class Input:
    def __init__(self, pic_path, start_year, end_year, predicted_year, days, history_days, predict_days):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize([128, 128]),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.3, 0.3, 0.3))
        ])
        self.pic_path = Path(pic_path)
        self.start_year = start_year
        self.end_year = end_year
        self.predicted_year = predicted_year
        self.days = days
        self.history_days = history_days
        self.predict_days = predict_days

    def load_image(self, year1, year2):
        images = []
        for year in range(year1, year2):
            temp_images = []
            temp_year_path = self.pic_path.joinpath(str(year))
            months = temp_year_path.joinpath('04')
            # for months in temp_year_path.iterdir():
            for file in months.iterdir():
                try:
                    image = plt.imread(str(file))
                    image = self.transform(image)
                    temp_images.append(image.numpy())
                except:
                    print(f'{file} is not added')
            images.append(temp_images)

        return images

    def image_data(self):
        """
        读取所有图像
        :return: self.images: 训练集图像  sel.predict_image： 测试集图像
        """

        train_images = self.load_image(self.start_year, self.end_year)
        test_images = self.load_image(self.predicted_year, self.predicted_year + 1)

        return train_images, test_images

    def splite(self, x):
        data = []
        label = []
        for i in range(len(x)):
            data.append(np.array(x[i][:29]))
            label.append(np.array(x[i][29:]))
        data = np.array(data)
        label = np.array(label)
        return np.array(data), np.array(label)

    def loader(self):
        train_data, test_data = self.image_data()
        train_x, train_y = self.splite(train_data)
        test_x, test_y = self.splite(test_data)
        train_loader = _loader(np.array(train_x), np.array(train_y))
        test_loader = _loader(np.array(test_x), np.array(test_y))
        return train_loader, test_loader
