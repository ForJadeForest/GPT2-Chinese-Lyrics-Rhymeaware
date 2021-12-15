import json
import os

import torch
from transformers import BertTokenizer

from model import GPT2LMHeadModel


def load_my_model(args):
    model_path = os.path.join(args.model_path, 'checkpoint-epoch={}'.format(args.epoch))
    finetune_path = os.path.join(args.model_path, 'fine_tune.json')
    device = torch.device(
        "cuda:{}".format(args.device) if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    with open(finetune_path, 'r') as f:
        finetune_args = json.load(f)
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True, never_split=['<word_space>'])
    model = GPT2LMHeadModel.from_pretrained(model_path, finetune_args=finetune_args)
    model.to(device)
    model.eval()
    return model, tokenizer
