import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm

from search_rhyme import get_sent_rhyme


def set_args():
    """设置模型预测所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, help='设置预测时使用的显卡,使用CPU设置成-1即可')
    parser.add_argument('--model_path',
                        default=r'/home/guest/yxyuan/Rap_generator/output_dir/[2021-12-14-13:11:11]中文歌词超级大模型', type=str,
                        help='模型文件路径')
    parser.add_argument('--vocab_path', default='./pre-train-model/model', type=str, help='词表，该词表为小词表，并增加了一些新的标记')
    parser.add_argument('--generate_max_len', default=256, type=int, help='生成标题的最大长度')
    parser.add_argument('--repetition_penalty', default=2, type=float, help='重复处罚率')
    parser.add_argument('--top_k', default=50, type=int, help='解码时保留概率最高的多少个标记')
    parser.add_argument('--top_p', default=0.95, type=float, help='解码时保留概率累加大于多少的标记')
    parser.add_argument('--max_len', type=int, default=512, help='输入模型的最大长度，要比config中n_ctx小')
    parser.add_argument('--epoch', type=int, default=27, help='所需模型的训练的次数')
    return parser.parse_args()


def top_k_top_p_filtering(logits, top_k, top_p, filter_value=-float("Inf")):
    """
    top_k或top_p解码策略，仅保留top_k个或累积概率到达top_p的标记，其他标记设为filter_value，后续在选取标记的过程中会取不到值设为无穷小。
    Args:
        logits: 预测结果，即预测成为词典中每个词的分数
        top_k: 只保留概率最高的top_k个标记
        top_p: 只保留概率累积达到top_p的标记
        filter_value: 过滤标记值

    Returns:

    """
    # logits的维度必须为，即size: vocab_size
    assert logits.dim() == 2
    # 获取top_k和字典大小中较小的一个，也就是说，如果top_k大于字典大小，则取字典大小个标记
    top_k = min(top_k, logits[0].size(-1))
    # 如果top_k不为0，则将在logits中保留top_k个标记
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    # 如果top_p不为0，则将在logits中保留概率值累积达到top_p的标记
    if top_p > 0.0:
        # 对logits进行递减排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        # 对排序后的结果使用softmax归一化，再获取累积概率序列
        # 例如：原始序列[0.1, 0.2, 0.3, 0.4]，则变为：[0.1, 0.3, 0.6, 1.0]
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # 删除累积概率高于top_p的标记
        sorted_indices_to_remove = cumulative_probs > top_p
        # 将索引向右移动，使第一个标记也保持在top_p之上
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0][indices_to_remove] = filter_value
    return logits


def predict_one_sample(model, tokenizer, device, args, content):
    input_ids, input_tensors, token_type_tensors, next_token_type, rhyme = text_process(content, tokenizer,
                                                                                        args.max_len,
                                                                                        args.generate_max_len, device)
    unk_id = tokenizer.convert_tokens_to_ids("[UNK]")
    sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
    generated = []
    # 用于存放，完成解码序列的序号
    finish_set = set()
    with torch.no_grad():
        # 遍历生成标题最大长度
        for _ in range(args.generate_max_len):

            outputs = model(input_ids=input_tensors, token_type_ids=token_type_tensors, rhyme_ids=rhyme)
            # 获取预测结果序列的最后一个标记，next_token_logits size：[batch_size, vocab_size]
            next_token_logits = outputs[0][:, -1, :]

            for token_id in set(generated):
                next_token_logits[0][token_id] /= args.repetition_penalty
            for last_word in input_ids[-3:-1]:
                next_token_logits[0][last_word] /= args.repetition_penalty
            # 惩罚上一句的末尾几个字
            # 对batch_size进行遍历，将词表中的UNK的值设为无穷小
            next_token_logits[0][unk_id] = -float("Inf")
            # 使用top_k_top_p_filtering函数，按照top_k和top_p的值，对预测结果进行筛选
            filter_logits = top_k_top_p_filtering(next_token_logits, top_k=args.top_k, top_p=args.top_p)
            # 对filter_logits的每一行做一次取值，输出结果是每一次取值时filter_logits对应行的下标，即词表位置（词的id）
            # filter_logits中的越大的值，越容易被选中
            next_tokens = torch.multinomial(F.softmax(filter_logits, dim=-1), num_samples=1)
            # 判断如果哪个序列的预测标记为sep_id时，则加入到finish_set

            if next_tokens[0, 0] == sep_id:
                break
            # 判断，如果finish_set包含全部的序列序号，则停止预测；否则继续预测
            # 将预测标记添加到generated中
            generated.append(next_tokens[0, 0].item())

            # 将预测结果拼接到input_tensors和token_type_tensors上，继续下一次预测
            input_tensors = torch.cat((input_tensors, next_tokens), dim=-1)
            token_type_tensors = torch.cat((token_type_tensors, next_token_type), dim=-1)
            next_tokens_char = tokenizer.convert_ids_to_tokens(next_tokens)
            rhyme = torch.cat((rhyme, torch.tensor([get_sent_rhyme(next_tokens_char)[0]]).to(device)), dim=-1)
    return "".join(tokenizer.convert_ids_to_tokens(generated)).replace("##", "").replace("<word_space>", " ")


def text_process(content, tokenizer, max_len, generate_max_len, device):
    content_tokens = tokenizer.tokenize(content.replace(" ", "<word_space>"))
    if len(content_tokens) > max_len - 3 - generate_max_len:
        content_tokens = content_tokens[:max_len - 3 - generate_max_len]
    content_id = 0
    title_id = 1

    # 将tokens索引化，变成模型所需格式
    content_tokens = ["[CLS]"] + content_tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(content_tokens)
    # 将input_ids和token_type_ids进行扩充，扩充到需要预测标题的个数，即batch_size
    token_type_ids = [content_id] * len(content_tokens)
    # 将input_ids和token_type_ids变成tensor
    input_tensors = torch.tensor([input_ids]).long().to(device)
    token_type_tensors = torch.tensor([token_type_ids]).long().to(device)
    next_token_type = torch.tensor([[title_id]]).long().to(device)
    rhyme = torch.tensor(get_sent_rhyme(content_tokens)).to(device)
    return input_ids, input_tensors, token_type_tensors, next_token_type, rhyme


def test_rhyme_acc(model, tokenizer, data, device, args):
    bar = tqdm(data, desc='Rhyme!:X.XXXXXX', ncols=100)
    right = 0
    num = 0
    for step, sent_pair in enumerate(bar):
        first = sent_pair['1']
        # 获取预测结果
        result = predict_one_sample(model, tokenizer, device, args, first)[0]
        if result == '' or first == '':
            continue
        if get_sent_rhyme(first)[-1] == get_sent_rhyme(result)[-1]:
            right += 1
        num += 1
        bar.set_description(desc='Rhyme!:{}'.format(round(right / num, 6)))
    return right, num
