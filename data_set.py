import torch
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset
import logging
from torch.nn.utils.rnn import pad_sequence
from search_rhyme import *
import random

logger = logging.getLogger(__name__)


def collate_func(batch_data):
    batch_size = len(batch_data)
    # 如果batch_size为0，则返回一个空字典
    if batch_size == 0:
        return {}
    input_ids_list, token_type_ids_list, rhyme_ids_list = [], [], []
    for instance in batch_data:
        # 按照batch中的最大数据长度,对数据进行padding填充
        input_ids_temp = instance["input_ids"]
        token_type_ids_temp = instance["token_type_ids"]
        rhyme_ids_temp = instance["rhyme_id"]
        # 将input_ids_temp和token_type_ids_temp添加到对应的list中
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        token_type_ids_list.append(torch.tensor(token_type_ids_temp, dtype=torch.long))
        rhyme_ids_list.append(torch.tensor(rhyme_ids_temp, dtype=torch.long))
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "token_type_ids": pad_sequence(token_type_ids_list, batch_first=True, padding_value=0),
            "rhyme_ids": pad_sequence(rhyme_ids_list, batch_first=True, padding_value=0)}


class GPT2RapGeneratorDataSet(Dataset):
    def __init__(self, tokenizer, max_len, first_max_len, data_dir, data_set_name, path_file=None, is_overwrite=False):
        self.tokenizer = tokenizer
        self.first_sent_id = 0
        self.second_sent_id = 1
        self.max_len = max_len
        self.first_max_len = first_max_len
        cached_feature_file = os.path.join(data_dir, "cached_{}_{}".format(data_set_name, max_len))
        # 判断缓存文件是否存在，如果存在，则直接加载处理后数据
        if os.path.exists(cached_feature_file) and not is_overwrite:
            logger.info("已经存在缓存文件{}，直接加载".format(cached_feature_file))
            self.data_set = torch.load(cached_feature_file)["data_set"]
        # 如果缓存数据不存在，则对原始数据进行数据处理操作，并将处理后的数据存成缓存文件
        else:
            logger.info("不存在缓存文件{}，进行数据预处理操作".format(cached_feature_file))
            self.data_set = self.load_data(path_file)
            logger.info("数据预处理操作完成，将处理后的数据存到{}中，作为缓存文件".format(cached_feature_file))
            torch.save({"data_set": self.data_set}, cached_feature_file)

    def load_data(self, path_file):
        """
        加载原始数据，生成数据处理后的数据
        Args:
            path_file: 原始数据路径

        Returns:

        """
        self.data_set = []
        with open(path_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            for idx, sample in enumerate(tqdm(data, desc="iter", disable=False)):
                input_ids, token_type_ids, rhyme_ids = self.convert_feature(sample)
                self.data_set.append({
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "rhyme_id": rhyme_ids
                })
            random.shuffle(self.data_set)
        return self.data_set

    def convert_feature(self, sample):
        input_ids = []
        token_type_ids = []
        rhyme_ids = []
        first_sent = self.tokenizer.tokenize(sample["1"].replace(" ", "<word_space>"))
        second_sent = self.tokenizer.tokenize(sample["2"].replace(" ", "<word_space>"))

        if len(first_sent) > self.first_max_len:
            first_sent = first_sent[:self.first_max_len]

        if len(second_sent) > self.max_len - len(first_sent) - 3:
            second_sent = second_sent[:self.max_len - len(first_sent) - 3]
        first_rhy = get_sent_rhyme(first_sent)
        second_rhy = get_sent_rhyme(second_sent)
        # 生成模型所需的input_ids和token_type_ids
        input_ids.append(self.tokenizer.cls_token_id)
        token_type_ids.append(self.first_sent_id)
        rhyme_ids.append(get_rhyme('<cls>'))

        input_ids.extend(self.tokenizer.convert_tokens_to_ids(first_sent))
        token_type_ids.extend([self.first_sent_id] * len(first_sent))
        rhyme_ids += first_rhy

        input_ids.append(self.tokenizer.sep_token_id)
        token_type_ids.append(self.first_sent_id)
        rhyme_ids.append(get_rhyme('<sep>'))

        input_ids.extend(self.tokenizer.convert_tokens_to_ids(second_sent))
        token_type_ids.extend([self.second_sent_id] * len(second_sent))
        rhyme_ids += second_rhy

        input_ids.append(self.tokenizer.sep_token_id)
        token_type_ids.append(self.second_sent_id)
        rhyme_ids.append(get_rhyme('<sep>'))

        # 判断input_ids长度是否小于等于最大长度
        assert len(input_ids) <= self.max_len

        return input_ids, token_type_ids, rhyme_ids

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        instance = self.data_set[idx]
        return instance
