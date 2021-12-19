import copy

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.models.gpt2 import GPT2PreTrainedModel, GPT2Model

from search_rhyme import *


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class Aggregator(nn.Module):
    def __init__(self, rhyme_embedding_size, word_embedding_size, fusion_dim, num_heads):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.fusion_matrix_rhyme = nn.Linear(rhyme_embedding_size, self.fusion_dim, bias=True)
        self.fusion_matrix_word = nn.Linear(word_embedding_size, self.fusion_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.rhyme_attention = nn.MultiheadAttention(embed_dim=rhyme_embedding_size, num_heads=num_heads, dropout=0.1,
                                                     batch_first=True)
        self.word_attention = nn.MultiheadAttention(embed_dim=word_embedding_size, num_heads=num_heads, dropout=0.1,
                                                    batch_first=True)
        self.back_rhyme = nn.Linear(self.fusion_dim, rhyme_embedding_size, bias=True)
        self.back_word = nn.Linear(self.fusion_dim, word_embedding_size, bias=True)

    def forward(self, word_seq_re, rhy_seq_re):
        device = word_seq_re.device
        mask = _generate_square_subsequent_mask(sz=word_seq_re.shape[1]).to(device)
        word_seq_re = self.word_attention(word_seq_re, word_seq_re, word_seq_re, attn_mask=mask)[0]
        rhy_seq_re = self.rhyme_attention(rhy_seq_re, rhy_seq_re, rhy_seq_re, attn_mask=mask)[0]
        h = self.relu(self.fusion_matrix_word(word_seq_re) + self.fusion_matrix_rhyme(rhy_seq_re))
        next_word_seq_re = self.relu(self.back_word(h))
        next_rhy_seq_re = self.relu(self.back_rhyme(h))
        return next_word_seq_re, next_rhy_seq_re


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = clones(LayerNorm(size), 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, word_re, rhyme_re, sublayer):
        "Apply residual connection to any sublayer with the same size."
        result = sublayer(self.norm[0](word_re), self.norm[1](rhyme_re))
        return word_re + self.dropout(result[0]), rhyme_re + self.dropout(result[1])


class GPT2LMHeadModel(GPT2PreTrainedModel):
    """GPT2模型"""

    def __init__(self, config, finetune_args=None):
        """
        初始化函数
        Args:
            config: 配置参数
            finetune_args: {
                fusion_dim : 韵脚和文字混合矩阵维度
                N:混合层的个数
                }
        """
        super().__init__(config)
        self.finetune_args = finetune_args
        self.N = finetune_args["N"]
        self.transformer = GPT2Model(config)
        self.has_res = finetune_args["has_res"]
        # 减少参数量，将rhyme的Embedding size设置与word Embedding一样
        rhyme_embedding_size = config.n_embd
        self.rhy_vocab = get_rhyme_vocab()
        self.fusion_dim = finetune_args["fusion_dim"]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.rhyme_embedding = nn.Embedding(len(self.rhy_vocab) + 1, rhyme_embedding_size, padding_idx=0)
        self.head_num = finetune_args["head_num"]
        self.aggregator = clones(
            Aggregator(rhyme_embedding_size, config.n_embd, self.fusion_dim, self.head_num),
            self.N)
        if self.has_res:
            self.sublayer = clones(SublayerConnection(size=config.n_embd, dropout=0.1), self.N)
        self.word_fusion = nn.Linear(config.n_embd,config.n_embd)
        self.rhyme_fusion = nn.Linear(config.n_embd, config.n_embd)
        self.relu = nn.ReLU(inplace=True)
        self.init_weights()

    def forward(self, input_ids=None, past=None, token_type_ids=None, labels=None, second_id=None, rhyme_ids=None):
        """
        前向函数，计算GPT2预测结果值
        Args:
            input_ids: 输入序列在词表中的索引序列，size:[batch_size, sequence_length]
            past: 包含由模型预先计算好的隐藏状态，一般使用在预测阶段，用于加速顺序解码，防止重复计算前面计算过的token
            token_type_ids: 用于区分输入序列中content和title的分隔符序列，size:[batch_size, sequence_length]
            labels: 标签序列，size:[batch_size, sequence_length]，一般情况下，与input_ids相同
            second_id: 第二句部分分隔符的id
            rhyme_ids: 韵脚ids 和Input_ids size相同
        Returns:

        """
        # 获取GPT2模型的输出结果
        rhy_embedding = self.rhyme_embedding(rhyme_ids)

        if len(rhy_embedding.shape) == 2:
            rhy_embedding = rhy_embedding.unsqueeze(0)
        transformer_outputs = self.transformer(input_ids, past_key_values=past, token_type_ids=token_type_ids)
        # 获取GPT2模型的最后一层的隐层节点状态，size:[batch_size, sequence_length, config.n_embd]
        hidden_states = transformer_outputs[0]
        word_represent = hidden_states
        if self.has_res:
            for layer, sublayer in zip(self.aggregator, self.sublayer):
                word_represent, rhy_embedding = sublayer(word_represent, rhy_embedding, lambda x, y: layer(x, y))
        else:
            for layer in self.aggregator:
                word_represent, rhy_embedding = layer(word_represent, rhy_embedding)
        # 预测隐层节点状态中的每一个token的下一个token，size:[batch_size, sequence_length, config.vocab_size]
        # hidden_states = word_represent + rhy_embedding
        hidden_states = self.relu(self.word_fusion(word_represent)) + self.relu(self.rhyme_fusion(rhy_embedding))

        lm_logits = self.lm_head(hidden_states)
        # 拼接输出结果
        # outputs = (lm_logits,) + transformer_outputs[1:]
        outputs = (lm_logits,)
        # 如果labels不为None时，计算损失值loss，并拼接到输出结果中
        if labels is not None:
            # 计算loss时，title_id不可以为None，因为需要title_id找到title的部分
            if second_id is None or token_type_ids is None:
                raise Exception("当labels不为None时， title_id和token_type_ids均不可以为None。")
            # 获取mask值，如果token_type_ids中等于title_id的部分需要计算loss，标记为1；否则为0。
            # size:[batch_size, sequence_length]
            mask = (token_type_ids == second_id).long()
            # 获取新的标签，size:[batch_size, sequence_length]
            labels = labels * mask
            # 对预测结果和标签进行偏移操作
            # GPT2的生成机制为通过前面的token，预测下一个token；并且labels与input_ids相同，
            # 因此input_ids中的第一个token的预测结果，实际上是标签中的第二个token，以此类推，最终仅计算sequence_length-1个token的loss
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # 定义损失函数CrossEntropyLoss，并且设置忽略计算loss的索引，以及返回loss的形式
            # 忽略shift_labels中为0的loss，也就是仅计算title部分的损失值
            # 对loss的计算方式设为sum，由于我们仅计算了itle部分的损失值，如果使用mean，会使loss变小（实际除的是sequence_length-1，不是title部分的真实长度）
            loss_fct = CrossEntropyLoss(ignore_index=0, reduction="sum")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # 获取title部分的真实长度，并计算真实loss
            num = shift_labels.ne(0).long().sum().item()
            loss = loss / num
            outputs = (loss,) + outputs
        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)

    @property
    def get_finetune_args(self):
        return self.finetune_args
