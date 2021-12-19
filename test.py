import json
import os

import torch

from utils import set_args, load_my_model, test_rhyme_acc

if __name__ == '__main__':
    args = set_args()
    model, tokenizer = load_my_model(args)
    with open('/home/guest/yxyuan/Rap_generator/data_dir/processing/big_data/test_data2.json', 'r',
              encoding='utf8') as f:
        data = json.loads(f.read())
    # 获取设备信息
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = args.device
    device = torch.device(
        "cuda:{}".format(args.device) if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    print(device)
    test_rhyme_acc(model, tokenizer, data, device, args)
