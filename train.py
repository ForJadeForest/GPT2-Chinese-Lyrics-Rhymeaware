import torch
import os
import random
import numpy as np
import argparse
import logging
import time
from transformers.models.gpt2 import GPT2Config
from model import GPT2LMHeadModel
from transformers import BertTokenizer
from data_set import collate_func, GPT2RapGeneratorDataSet
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(model, device, train_data, test_data, args):
    """
    训练模型
    Args:
        model: 模型
        device: 设备信息
        train_data: 训练数据类
        test_data: 测试数据类
        args: 训练参数配置信息

    Returns:

    """

    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps参数无效，必须大于等于1")
    # 计算真实的训练batch_size大小
    train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size,
                                   collate_fn=collate_func, num_workers=8, pin_memory=False)
    total_steps = int(len(train_data_loader) * args.num_train_epochs / args.gradient_accumulation_steps)
    logger.info("总训练步数为:{}".format(total_steps))
    model.to(device)
    # 获取模型所有参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # 设置优化器
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    # 清空cuda缓存
    torch.cuda.empty_cache()
    # 将模型调至训练状态
    model.train()
    second_sent_id = train_data.second_sent_id
    tr_loss, logging_loss, min_loss = 0.0, 0.0, 0.0
    global_step = 0
    time_list = []

    now_time = get_time().replace(' ', '-')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    output_dir = os.path.join(args.output_dir, '[' + now_time + ']' + args.simple_desc)
    finetune_para_path = os.path.join(output_dir, 'fine_tune.json')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(output_dir, 'desc.txt'), 'w') as f:
        f.write('=' * 100 + '\n')
        for k in args.__dict__:
            f.write(str(k) + ": " + str(args.__dict__[k]) + '\n')
        f.write('=' * 100 + '\n')
        model_text = model.__repr__()
        f.write(model_text[:16] + model_text[6526:])

    with open(finetune_para_path, 'w') as f:
        data2 = json.dumps(model.get_finetune_args, sort_keys=True, indent=4, separators=(',', ': '))
        f.write(data2)

    summary_path = os.path.join(output_dir, 'runs')
    tb_write = SummaryWriter(summary_path)
    # 开始训练模型
    for iepoch in range(0, int(args.num_train_epochs)):
        iter_bar = tqdm(train_data_loader, position=0, desc="Iter (loss=X.XXX)", disable=False, ncols=100)
        # start = time.time()
        for step, batch in enumerate(iter_bar):
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            rhyme_ids = batch['rhyme_ids'].to(device)
            # 获取训练结果
            outputs = model.forward(input_ids=input_ids, token_type_ids=token_type_ids, labels=input_ids,
                                    second_id=second_sent_id, rhyme_ids=rhyme_ids)
            loss = outputs[0]
            tr_loss += loss.item()
            # 将损失值放到Iter中，方便观察
            iter_bar.set_description("Epoch:{}, Iter (loss=%5.3f)".format(iepoch) % loss.item())
            # 判断是否进行梯度累积，如果进行，则将损失值除以累积步数
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            # 损失进行回传
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # 当训练步数整除累积步数时，进行参数优化
            if (step + 1) % args.gradient_accumulation_steps == 0:

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                # 如果步数整除logging_steps，则记录学习率和训练集损失值
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_write.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_write.add_scalar("train_loss", (tr_loss - logging_loss) /
                                        (args.logging_steps * args.gradient_accumulation_steps), global_step)
                    logging_loss = tr_loss
                # 如果步数整除eval_steps，则进行模型测试，记录测试集的损失
                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    eval_loss = evaluate(model, device, test_data, args)
                    tb_write.add_scalar("test_loss", eval_loss, global_step)
                    model.train()


        # 每个epoch进行完，则保存模型
        # end_time = time.time()
        # time_list.append((end_time - start))
        model_output_dir = os.path.join(output_dir, "checkpoint-epoch={}".format(iepoch))
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(model_output_dir)
        # 清空cuda缓存
        torch.cuda.empty_cache()
    with open(os.path.join(output_dir, 'desc.txt'), 'a') as f:
        f.write("average_epoch_time: {} s".format(sum(time_list) / len(time_list)) + '\n')
        f.write("all_epoch_time: {} s".format(sum(time_list)) + '\n')
        f.write("模型描述：" + '\n' + args.desc + '\n')


def get_time():
    # 获得当前时间时间戳
    now = int(time.time())
    # 转换为其他日期格式,如:"%Y-%m-%d %H:%M:%S"
    timeArray = time.localtime(now)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime


def evaluate(model, device, test_data, args):
    """
    对测试数据集进行模型测试
    Args:
        model: 模型
        device: 设备信息
        test_data: 测试数据类
        args: 训练参数配置信息

    Returns:

    """
    # 构造测试集的DataLoader
    print("开始测试阶段")
    test_sampler = SequentialSampler(test_data)
    test_data_loader = DataLoader(test_data, sampler=test_sampler,
                                  batch_size=args.test_batch_size, collate_fn=collate_func)
    iter_bar = tqdm(test_data_loader, position=1, desc="test iter", disable=False, ncols=100)
    second_id = test_data.second_sent_id
    total_loss, total = 0.0, 0.0
    # 进行测试
    for step, batch in enumerate(iter_bar):
        # 模型设为eval
        model.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            rhyme_ids = batch['rhyme_ids'].to(device)
            # 获取预测结果
            outputs = model.forward(input_ids=input_ids, token_type_ids=token_type_ids, labels=input_ids,
                                    second_id=second_id, rhyme_ids=rhyme_ids)
            loss = outputs[0]
            loss = loss.item()
            # 对loss进行累加
            total_loss += loss * len(batch["input_ids"])
            total += len(batch["input_ids"])

    # 计算最终测试集的loss结果
    test_loss = total_loss / total
    print("结束测试，损失为{}".format(test_loss))
    return test_loss


def set_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='设置训练或测试时使用的显卡')
    parser.add_argument('--config_path', default='./model/config.json', type=str, help='模型参数配置信息')
    parser.add_argument('--vocab_path', default='./model/vocab.txt', type=str, help='词表，该词表为小词表，并增加了一些新的标记')
    parser.add_argument('--train_file_path', default='./data_dir/rap_data.json', type=str, help='新闻标题生成的训练数据')
    parser.add_argument('--test_file_path', default='./data_dir/rap_data.json', type=str, help='新闻标题生成的测试数据')
    parser.add_argument('--pretrained_model_path', default='./model', type=str, help='预训练的GPT2模型的路径')
    parser.add_argument('--data_dir', default='./data_dir', type=str, help='生成缓存数据的存放路径')
    parser.add_argument('--num_train_epochs', default=2, type=int, help='模型训练的轮数')
    parser.add_argument('--train_batch_size', default=64, type=int, help='训练时每个batch的大小')
    parser.add_argument('--test_batch_size', default=64, type=int, help='测试时每个batch的大小')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='模型训练时的学习率')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='warm up概率，即训练总步长的百分之多少，进行warm up')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Adam优化器的epsilon值')
    parser.add_argument('--logging_steps', default=20, type=int, help='保存训练日志的步数')
    parser.add_argument('--eval_steps', default=1000, type=int, help='训练时，多少步进行一次测试')
    parser.add_argument('--gradient_accumulation_steps', default=2, type=int, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--output_dir', default='output_dir/', type=str, help='模型输出路径')
    parser.add_argument('--seed', type=int, default=2020, help='随机种子')
    parser.add_argument('--max_len', type=int, default=512, help='输入模型的最大长度，要比config中n_ctx小')
    parser.add_argument('--first_max_len', type=int, default=32, help='生成标题的最大长度，要比max_len小')
    parser.add_argument('--Aggregator_num', type=int, default=6, help='Aggregator层数量')
    parser.add_argument('--fusion_dim', type=int, default=768, help='韵律embedding和词语embedding混合后的维度')
    parser.add_argument('--has_res', type=bool, default=False, help='是否使用残差')
    parser.add_argument('--desc', type=str, default=None, help='训练描述')
    parser.add_argument('--simple_desc', type=str, default=None, help='训练描述')

    return parser.parse_args()


def main():
    # 设置模型训练参数

    args = set_args()
    torch.multiprocessing.set_start_method('spawn')
    while not os.path.exists(args.train_file_path):
        time.sleep(120)
        continue

    fine_tune_args = {
        'N': args.Aggregator_num,
        'fusion_dim': args.fusion_dim,
        'has_res': args.has_res,
    }
    # 设置显卡信息
    print('=' * 100)
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    print('=' * 100)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = args.device
    # 获取device信息，用于模型训练
    device = torch.device(
        "cuda:{}".format(int(args.device)) if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    # 设置随机种子，方便模型复现
    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    # 加载模型的config
    model_config = GPT2Config.from_json_file(args.config_path)
    if args.pretrained_model_path:
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model_path, finetune_args=fine_tune_args)
    else:
        # 如果没有指定的预训练模型，则初始化模型
        model = GPT2LMHeadModel(config=model_config, finetune_args=fine_tune_args)
    # 实例化tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True, do_basic_tokenize=True,
                                              never_split=['<word_space>'])
    # 创建模型的输出目录

    # 加载训练数据和测试数据
    train_data = GPT2RapGeneratorDataSet(tokenizer, args.max_len, args.first_max_len, args.data_dir, "train",
                                         args.train_file_path)
    test_data = GPT2RapGeneratorDataSet(tokenizer, args.max_len, args.first_max_len, args.data_dir, "test",
                                        args.test_file_path)
    # 开始训练
    train(model, device, train_data, test_data, args)


if __name__ == '__main__':
    main()
