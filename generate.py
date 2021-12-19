import argparse
import json
import os

import torch
from transformers import BertTokenizer

from model import GPT2LMHeadModel
from utils import predict_one_sample


def set_args():
    """设置模型预测所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='-1', type=str, help='设置预测时使用的显卡,使用CPU设置成-1即可')
    parser.add_argument('--model_path',
                        default=r'.\pre-train-model\final_model', type=str,
                        help='模型文件路径')
    parser.add_argument('--vocab_path', default=r'.\pre-train-model\final_model', type=str,
                        help='词表，该词表为小词表，并增加了一些新的标记')
    parser.add_argument('--generate_max_len', default=256, type=int, help='生成标题的最大长度')
    parser.add_argument('--repetition_penalty', default=2, type=float, help='重复处罚率')
    parser.add_argument('--top_k', default=50, type=int, help='解码时保留概率最高的多少个标记')
    parser.add_argument('--top_p', default=0.95, type=float, help='解码时保留概率累加大于多少的标记')
    parser.add_argument('--max_len', type=int, default=512, help='输入模型的最大长度，要比config中n_ctx小')
    return parser.parse_args()


def main():
    args = set_args()
    # 获取设备信息
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = args.device
    device = torch.device(
        "cuda:{}".format(args.device) if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    print(device)
    finetune_path = os.path.join(args.model_path, 'fine_tune.json')
    with open(finetune_path, 'r') as f:
        finetune_args = json.load(f)
    content = input("输入第一句歌词")
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True, never_split=['<word_space>'])
    model = GPT2LMHeadModel.from_pretrained(args.model_path, finetune_args=finetune_args)
    model.to(device)
    model.eval()
    while content:
        for _ in range(10):
            next = predict_one_sample(model, tokenizer, device, args, content)
            print(next)
            content = content + '[SEP]' + next
        print("=" * 20 + "生成完毕" + "=" * 20)
        content = input("输入下一句歌词")


if __name__ == '__main__':
    main()
