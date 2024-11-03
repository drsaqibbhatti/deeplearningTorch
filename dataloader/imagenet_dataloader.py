
from dataset.imagenet_dataset import imagenet_dataset
import random
import torch

class imagenet_dataloader():
    def __init__(self, dataset:imagenet_dataset, batch_size = 1, shuffle=True, drop_last=True):
        super(imagenet_dataloader, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.batch_index = -1
        self.dataset_length = dataset.__len__()
        self.batch_total = self.dataset_length // self.batch_size
        self.batch_list = [i for i in range(self.dataset_length)]
        self.batch_dataset = []
        if self.shuffle == True:
            random.shuffle(self.batch_list)

        for batch_index in range(self.batch_total):
            batch = []
            for batch_each in range(self.batch_size):
                batch.append(self.batch_list[batch_each + batch_index * self.batch_size])
            self.batch_dataset.append(batch)

        if self.drop_last == False:
            batch = []
            for batch_each in range(self.batch_total*self.batch_size, self.dataset_length):
                batch.append(self.batch_list[batch_each])
            self.batch_dataset.append(batch)
            self.batch_total += 1


    def __iter__(self):
        return self

    def __len__(self):
        return self.batch_total

    def __next__(self):
        self.batch_index += 1
        if self.batch_index < 0 or self.batch_index > self.batch_total:
            self.batch_index = 0
            self.batch_dataset.clear()
            self.batch_total = self.dataset_length // self.batch_size
            if self.shuffle == True:
                random.shuffle(self.batch_list)

            for batch_index in range(self.batch_total):
                batch = []
                for batch_each in range(self.batch_size):
                    batch.append(self.batch_list[batch_each + batch_index * self.batch_size])
                self.batch_dataset.append(batch)

            if self.drop_last == False:
                batch = []
                for batch_each in range(self.batch_total * self.batch_size, self.dataset_length):
                    batch.append(self.batch_list[batch_each])
                self.batch_dataset.append(batch)
                self.batch_total += 1

        batch = self.batch_dataset[self.batch_index]
        images = torch.tensor([])
        hitmaps = torch.tensor([])
        for index in batch:
            image, hitmap = self.dataset.__getitem__(index)
            images = torch.cat((images, image.unsqueeze(dim=0)), dim=0)
            hitmaps = torch.cat((hitmaps, hitmap.unsqueeze(dim=0)),dim=0)
        return images, hitmaps