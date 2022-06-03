# GPT2-Chinese-Lyrics-Rhymeaware

make lyrics generated more rhyme!

### Description

- 本模型旨在生成更具有韵律的歌词语句。在基于GPT2模型的基础上加入歌词信息与韵律的混合层从而进行微调。
- 本项目使用的训练集来自[这里](https://github.com/gaussic/Chinese-Lyric-Corpus)，为了进一步方便模型对韵脚的学习能力。提取了一首歌中相互押韵的句子对。也就是说将歌词每两句组成一对，再剔除不押韵的句子对。
- 本项目使用的预训练模型来自 [Cheng hou](https://github.com/hhou435)训练的中文歌词训练模型。可以从 [谷歌云盘](https://drive.google.com/drive/folders/1RFq4NoQ3phCJjrhKtu2Xbn6z0krcN9TM?usp=sharing)和 [百度云盘（0qnn）](https://pan.baidu.com/s/19x0d0bPGCWHi9L4Pu0pSiw)处下载。更多信息参考 [GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese)项目。

### 代码结构介绍

- generate.py 与 train.py 分别是样例测试与训练的脚本。
- model.py 保存模型代码结构
- data_set.py 保存pytorch的dataset类
- search.py 保存获取韵律字典以及获取一句话的韵脚等函数
- test.py 用于检测训练好的模型押韵的准确率。
- utils.py 包含一些测试函数以及加载模型函数，用于训练后检测效果如何
- data_process.py用于处理下载好的数据集。生成相应的文件。如果你想使用别的数据集，请处理成相应的格式。已经提供样例。
- 其他
  - rhyme_finals.txt 保存了互相押韵的韵脚，每一行表示相互押韵。基础来自[Rap generator](https://github.com/Hongyu-Li/RapGenerator_GPT2)，个人在进行了一些微小的调整
  - output_dir 文件夹保存了训练好的模型
  - final_model 文件夹存放最终的模型，如果你只想体验模型，下载我训练好的模型放入该文件夹下即可用于查看效果。
  - pre-train-model 文件夹保存别人预训练好的模型。
  - data_dir 文件夹存放数据。其中包含两个文件夹，一个是原始数据的文件夹，一个是处理好之后的文件夹

### model

- 受启发于 [ERNIE: Enhanced Language Representation with Informative Entities](https://arxiv.org/abs/1905.07129) 知识融合方式，项目中设计了韵律融合结构，如下图
  - <img src="https://jadepicgo.oss-cn-shenzhen.aliyuncs.com/img/art.png" alt="art" style="zoom:50%;" />


  - <img src="https://jadepicgo.oss-cn-shenzhen.aliyuncs.com/img/net1.jpg" alt="net1" style="zoom:50%;" />
- 文本先通过GPT2获得相应的序列表示，在经过多个Aggregator层进行韵律和文本的融合。在最后一层再经过一个线性映射层得到相应的logits。输出结果。



### Trian

- 训练模型需要设置一些参数，以下是参数介绍

|        训练参数设置         |                 相关描述                 |
| :-------------------------: | :--------------------------------------: |
|           device            |        设置训练或测试时使用的显卡        |
|         config_path         |           模型参数配置信息路径           |
|         vocab_path          |                 词表路径                 |
|       train_file_path       |              训练集数据路径              |
|       test_file_path        |              测试集数据路径              |
|    pretrained_model_path    |              预训练模型路径              |
|          data_dir           |          生成缓存数据的存放路径          |
|      num_train_epochs       |              模型训练的轮数              |
|      train_batch_size       |          训练时每个batch的大小           |
|       test_batch_size       |          测试时每个batch的大小           |
|        learning_rate        |            模型训练时的学习率            |
|      warmup_proportion      |               warm up概率                |
|        adam_epsilon         |          Adam优化器的epsilon值           |
|        logging_steps        |            保存训练日志的步数            |
|         eval_steps          |        训练时，多少步进行一次测试        |
| gradient_accumulation_steps |               梯度积累步数               |
|        max_grad_norm        |                梯度正则化                |
|         output_dir          |               模型输出路径               |
|            seed             |                 随机种子                 |
|           max_len           |            输入模型的最大长度            |
|       second_max_len        |            生成歌词的最大长度            |
|       Aggregator_num        |             Aggregator层数量             |
|         fusion_dim          | 韵律embedding和词语embedding混合后的维度 |
|           has_res           |               是否使用残差               |
|            desc             |            训练描述，记录信息            |
|         simple_desc         |   训练描述，会附加在模型保存文件夹名后   |
|          head_num           |              注意力头的个数              |
|       continue_train        |             是否继续训练模型             |
|            epoch            |  从第几轮开始训练，默认为-1，即重头开始  |

- 在命令行输入如下参数即可训练

```sh
python train.py --device=0 --num_train_epochs=50 --train_batch_size=128 --test_batch_size=32 --learning_rate=5e-6 --warmup_proportion=0.1 --adam_epsilon=1e-5 --logging_steps=20 --eval_steps=2000 --gradient_accumulation_steps=4 --max_grad_norm=1.0 --output_dir=./output_dir --seed=2021 --max_len=512 --first_max_len=256 --config_path=./pre-train-model/model/config.json --vocab_path=./pre-train-model/model --pretrained_model_path=./pre-train-model/model --train_file_path=./data_dir/processing/all_lyrics/train_data.json --test_file_path=./data_dir/processing/all_lyrics/test_data.json --data_dir=./data_dir/processing/all_lyrics --Aggregator_num=3 --has_res=True --head_num=4 --fusion_dim=512 --simple_desc=样例 --epoch=-1 --desc=这是一个训练样例
```

- 训练结果会默认保存在`./output_dir/[模型开始训练的时间]+simple_desc`中，其中包含每一个epoch训练结果和tensorboard日志记录文件，`fine_tune.json`（包含模型参数设置信息），以及一个`desc.txt`记录训练参数以及模型架构文件。



### Generate

- 直接运行generate.py文件即可查看测试结果

```sh
python generate.py --device=0 --model_path=.\final_model --vocab_path=.\final_model --top_k=50 --top_p=0.95
```

- 测试结果，采用`top_k=50, top_p=0.95`保证生成的语句多样性
  - <img src="https://jadepicgo.oss-cn-shenzhen.aliyuncs.com/img/image-20211219151104419.png" alt="image-20211219151104419" style="zoom:50%;" />
  - <img src="https://jadepicgo.oss-cn-shenzhen.aliyuncs.com/img/image-20211219151538672.png" alt="image-20211219151538672" style="zoom:50%;" />

### model download

- [谷歌网盘](https://drive.google.com/drive/folders/1SBRP7WEctNnV0puUz69VwH3AYAHzQEm8?usp=sharing)

### Acknowledge

- 本项目中部分训练代码与生成来自于[GPT2-NewsTitle](https://github.com/liucongg/GPT2-NewsTitle)，其启发我使用句子对的方式进行训练。
- 同时项目灵感来源于 [RapGenerator_GPT2](https://github.com/Hongyu-Li/RapGenerator_GPT2)，该项目采用滚回的方式来强制押韵。但作为rap爱好者，还是希望能够让模型自动生成一些带韵脚的歌词。

