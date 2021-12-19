import json
import os
import random
import re
import time

from tqdm import tqdm

from search_rhyme import *


def satisfy_condition(lyrics_list, first, second, rap_dict):
    if first == '' or second == '':
        return False
    if rap_dict in lyrics_list:
        return False
    if first[-1] == second[-1]:
        return False
    if get_sent_rhyme(first)[-1] != get_sent_rhyme(second)[-1]:
        return False
    if first == second:
        return False
    if len(first) < 3 or len(second) < 3:
        return False
    return True


def Chinese_Lyrics(data_path, save_path):
    lyrics_list = []
    data = []
    singer_bar = tqdm(os.listdir(data_path), position=0, desc='Singer', ncols=80, leave=False)
    for singer in singer_bar:
        singer_name = re.findall(r'(.*)_[0-9]*', singer)[0]
        singer_path = os.path.join(data_path, singer)
        singer_bar.set_description("Singer {}".format(singer))
        sent_num = 0
        all_num = 0
        start_time = time.time()
        for lyrics in os.listdir(singer_path):
            lyrics_path = os.path.join(singer_path, lyrics)
            with open(lyrics_path, 'r', encoding='utf8') as f:
                context_list = f.readlines()
                for first, second in zip(context_list[::2], context_list[1::2]):
                    first = first.strip()
                    second = second.strip()
                    first = first.replace('\n', '')
                    second = second.replace('\n', '')
                    first = first.replace('(', '')
                    second = second.replace('(', '')
                    first = first.replace(')', '')
                    second = second.replace(')', '')
                    first = first.replace(' ', ' ')
                    second = second.replace(' ', ' ')
                    if re.findall('（.*）(.*)', first):
                        first = re.findall('（.*）(.*)', first)[0]
                    if re.findall('（.*）(.*)', second):
                        second = re.findall('（.*）(.*)', second)[0]
                    rap_dict = {
                        '1': first,
                        '2': second
                    }
                    all_num += 1
                    if satisfy_condition(lyrics_list, first, second, rap_dict):
                        lyrics_list.append(rap_dict)
                        sent_num += 1
                    else:
                        continue
        ratio = sent_num / all_num
        data_dict = {
            'ratio': ratio,
            'sent_num': sent_num,
            'all_num': all_num,
            'singer': singer_name,
            'time': time.time() - start_time
        }

        data.append(data_dict)
        singer_bar.write('{}共收集到{}个句子，占比{}，花费 {} s'.format(singer_name, sent_num, sent_num / all_num,
                                                           round(time.time() - start_time)))

    lyrice_path = os.path.join(save_path, 'Chinese_Lyrics.json')
    with open(lyrice_path, 'w', encoding='utf8') as w_f:
        w_f.write(json.dumps(lyrics_list, ensure_ascii=False, indent=4))
    p = os.path.join(save_path, 'Chinese_Lyrics_data.json')
    with open(p, 'w', encoding='utf8') as w_f:
        w_f.write(json.dumps(data, ensure_ascii=False, indent=4))


data_path = '/home/guest/yxyuan/Rap_generator/Chinese_Lyrics/Chinese_Lyrics'
save_path = '/home/guest/yxyuan/Rap_generator/data_dir/processing/big_data'
Chinese_Lyrics(data_path, save_path)
train_len = 0.9
test_len = 0.1
train_data = []
test_data = []
train_save = os.path.join(save_path,'train_data.json')
test_save = os.path.join(save_path,'test_data.json')
with open(os.path.join(save_path, 'Chinese_Lyrics.json'), 'r', encoding='utf8')as f:
    text = ''.join(f.readlines())
    dict = json.loads(text)
    random.shuffle(dict)
    for i in tqdm(range(len(dict))):
        if i <= len(dict) * train_len:
            train_data.append(dict[i])
        else:
            test_data.append(dict[i])
    with open(train_save, 'w', encoding='utf8')as f_train:
        f_train.write(json.dumps(train_data,ensure_ascii=False, indent=4))
    with open(test_save, 'w', encoding='utf8')as f_test:
        f_test.write(json.dumps(test_data,ensure_ascii=False, indent=4))