from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from model.clip_model.simple_tokenizer import SimpleTokenizer
import os
import numpy as np
import scipy.io as scio

from torch.utils.data import Dataset
import torch
import random
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


class BaseDataset(Dataset):
    def __init__(self,
                 captions: dict,
                 indexs: dict,
                 labels: dict,
                 is_train=True,
                 tokenizer=SimpleTokenizer(),
                 maxWords=32,
                 imageResolution=224,
                 ):

        self.captions = captions
        self.indexs = indexs
        self.labels = labels
        self.maxWords = maxWords
        self.tokenizer = tokenizer

        self.transform = Compose([
            Resize(imageResolution, interpolation=Image.BICUBIC),
            CenterCrop(imageResolution),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]) if is_train else Compose([
            Resize((imageResolution, imageResolution), interpolation=Image.BICUBIC),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

        self.__length = len(self.indexs)

    def __len__(self):
        return self.__length

    def _load_image(self, index: int) -> torch.Tensor:
        image_path = self.indexs[index].strip()
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image

    def _load_text(self, index: int):
        captions = self.captions[index]
        words = self.tokenizer.tokenize(captions)
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.maxWords - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        caption = self.tokenizer.convert_tokens_to_ids(words)
        while len(caption) < self.maxWords:
            caption.append(0)
        caption = torch.tensor(caption)
        key_padding_mask = (caption == 0)
        return caption, key_padding_mask

    def _load_label(self, index: int) -> torch.Tensor:
        label = self.labels[index]
        label = torch.from_numpy(label)
        return label

    def get_all_label(self):
        labels = torch.zeros([self.__length, len(self.labels[0])], dtype=torch.int64)
        for i, item in enumerate(self.labels):
            labels[i] = torch.from_numpy(item)
        return labels

    def __getitem__(self, index):
        image = self._load_image(index)
        caption, key_padding_mask = self._load_text(index)
        label = self._load_label(index)
        return image, caption, key_padding_mask, label, index


def split_data(captions, indexs, labels, query_num, train_num, seed=None):

    np.random.seed(seed=1)  # fixed to 1 for all experiments.

    random_index = np.random.permutation(range(len(indexs)))
    query_index = random_index[: query_num]
    train_index = random_index[query_num: query_num + train_num]
    retrieval_index = random_index[query_num:]

    query_indexs = indexs[query_index]
    query_captions = captions[query_index]
    query_labels = labels[query_index]

    train_indexs = indexs[train_index]
    train_captions = captions[train_index]
    train_labels = labels[train_index]

    retrieval_indexs = indexs[retrieval_index]
    retrieval_captions = captions[retrieval_index]
    retrieval_labels = labels[retrieval_index]

    split_indexs = (query_indexs, train_indexs, retrieval_indexs)
    split_captions = (query_captions, train_captions, retrieval_captions)
    split_labels = (query_labels, train_labels, retrieval_labels)

    return split_indexs, split_captions, split_labels


def generate_dataset(captionFile: str,
                     indexFile: str,
                     labelFile: str,
                     maxWords=32,
                     imageResolution=224,
                     query_num=2000,
                     train_num=10000,
                     seed=None,
                     ):

    captions = scio.loadmat(captionFile)["caption"]
    indexs = scio.loadmat(indexFile)["index"]
    labels = scio.loadmat(labelFile)["label"]

    split_indexs, split_captions, split_labels = split_data(captions, indexs, labels, query_num=query_num, train_num=train_num, seed=seed)

    query_data = BaseDataset(captions=split_captions[0], indexs=split_indexs[0], labels=split_labels[0],
                             maxWords=maxWords, imageResolution=imageResolution, is_train=False)
    train_data = BaseDataset(captions=split_captions[1], indexs=split_indexs[1], labels=split_labels[1],
                             maxWords=maxWords, imageResolution=imageResolution)
    retrieval_data = BaseDataset(captions=split_captions[2], indexs=split_indexs[2], labels=split_labels[2],
                                 maxWords=maxWords, imageResolution=imageResolution, is_train=False)

    return train_data, query_data, retrieval_data

