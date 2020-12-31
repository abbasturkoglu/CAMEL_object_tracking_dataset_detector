from functools import partial
import os
import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt

import torch
from PIL import Image


def create_data_df(data_dir):

    directories = [name for name in os.listdir(data_dir) if name.startswith('sequence')]

    all_data_df = None

    for n, sequence in enumerate(sorted(directories), 1):

        frames = os.listdir(os.path.join(data_dir, sequence, 'img1'))
        frames = sorted(frames)

        annotations = pd.read_csv(os.path.join(data_dir, sequence, 'Seq'+str(n)+'-Vis.txt'), sep='\t', header=None)
        frame_annotations = annotations.groupby([0]).apply(lambda x: x.values)
        frame_annotations = frame_annotations.reindex(range(1,max(frame_annotations.index)+1))

        sequence_df = pd.DataFrame({'annotation':frame_annotations, 'img_name': frames, 'sequence': [n]*len(frames), 'dir_name': [sequence]*len(frames)})
        sequence_df.index.name = 'frame_num'
        sequence_df = sequence_df.reset_index()

        if all_data_df is not None:
            all_data_df = pd.concat([all_data_df, sequence_df])
        else:
            all_data_df = sequence_df

    all_data_df = all_data_df.reset_index(drop=True)

    return all_data_df

def create_train_test_split(data_df, size):

    def _is_in_train(data, size):
        return data.frame_num < len(data)*size

    is_in_train = partial(_is_in_train, size=size)

    data_df['train'] = data_df.groupby(['sequence']).apply(is_in_train).reset_index()['frame_num']

    return data_df


class PeopleDataset(object):

    def __init__(self, transforms, data_df=None, data_root='../data'):
        self.transforms = transforms
        self.data_root = data_root
        if data_df is None:
            self.data_df = create_data_df(data_root)
        else:
            self.data_df = data_df

    def __getitem__(self, idx):
        # load images ad masks
        item_data = self.data_df.iloc[idx]
        img_path = os.path.join(self.data_root, item_data['dir_name'], 'img1', item_data['img_name'])
        img = Image.open(img_path).convert("RGB")

        # get bounding box coordinates for each mask
        if type(item_data['annotation']) is not float:
            num_objs = item_data['annotation'].shape[0]
        else:
            num_objs = 0

        boxes = []

        for i in range(num_objs):
            pos = item_data['annotation'][i, 3:]
            xmin = pos[0]
            xmax = pos[0]+pos[2]
            ymin = pos[1]
            ymax = pos[1]+pos[3]
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        if num_objs!=0:
            labels = torch.as_tensor(item_data['annotation'][:, 2].tolist(), dtype=torch.int64)
        else:
            labels = torch.as_tensor([], dtype=torch.int64)

        image_id = torch.tensor([idx])

        if num_objs:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = 0

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = item_data['img_name']
        target["area"] = area

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return self.data_df.shape[0]

    def train_test_split(self, size):

        data_df = create_train_test_split(self.data_df, size)
        train_df = data_df.loc[data_df['train']==True]
        test_df = data_df.loc[data_df['train']==False]

        return PeopleDataset(self.transforms, train_df, data_root=self.data_root), PeopleDataset(self.transforms, test_df, data_root=self.data_root)
