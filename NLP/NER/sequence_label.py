# !/usr/bin/env python
# -*- coding:utf-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning on sequence labeling task."""

import argparse
import ast

import paddle.fluid as fluid
import paddlehub as hub

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.0, help="Warmup proportion params for warmup strategy")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--use_pyreader", type=ast.literal_eval, default=False, help="Whether use pyreader to feed data.")
parser.add_argument("--use_data_parallel", type=ast.literal_eval, default=False, help="Whether use data parallel.")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    # Step1:加载ERNIE预训练模型（主要的变动就是这里加上了版本号码）
    module = hub.Module(name="ernie", version="1.0.2")
    '''
    :param name:{ernie, bert_uncased_L-12_H-768_A-12, etc}
    '''
    inputs, outputs, program = module.context(trainable=True, max_seq_len=128)
    # 更换name参数即可无缝切换BERT中文模型, 代码示例如下
    # module = hub.Module(name="bert_chinese_L-12_H-768_A-12")inputs, outputs, program = module.context(max_seq_len=128)

    # Step2: 准备数据集并使用SequenceLabelReader读取数据
    # 可以通过自定义一个数据集的类
    dataSet = hub.dataset.MSRA_NER()  # 其中数据集的准备代码可以参考 msra_ner.py
    reader = hub.reader.SequenceLabelReader(
        dataset=dataSet,
        vocab_path=module.get_vocab_path(),
        max_seq_len=128)
    '''
    :param hub.dataset.MSRA_NER() 会自动从网络下载数据集并解压到用户目录下$HOME/.paddlehub/dataset目录
    :param module.get_vaocab_path() 会返回预训练模型对应的词表
    :param max_seq_len 需要与Step1中context接口传入的序列长度保持一致
    :param SequenceLabelReader中的data_generator会自动按照模型对应词表对数据进行切词，以迭代器的方式返回ERNIE/BERT所需要的Tensor格式，包括input_ids，position_ids，segment_id与序列对应的mask input_mask(输出顺序是默认的)
    '''

    # Step3：选择优化策略和运行配置
    strategy = hub.AdamWeightDecayStrategy(
        weight_decay=0.01,
        warmup_proportion=0.1,
        learning_rate=5e-5,
        lr_scheduler="linear_decay",
        optimizer_name="adam")
    '''
    :param learning_rate: Finetune过程中的最大学习率
    :param weight_decay: 模型的正则项参数，默认0.01，如果模型有过拟合倾向，可适当调高这一参数
    :param warmup_proportion: 如果warmup_proportion>0, 例如0.1, 则学习率会在前10%的steps中线性增长至最高值learning_rate
    :param lr_scheduler: 有两种策略可选(1) linear_decay策略学习率会在最高点后以线性方式衰减; noam_decay策略学习率会在最高点以多项式形式衰减
    '''

    # Setup runing config for PaddleHub Finetune API
    config = hub.RunConfig(
        use_data_parallel=False,
        use_pyreader=False,
        use_cuda=False,
        batch_size=1,
        enable_memory_optim=False,
        checkpoint_dir='ernie_seq_label_turtorial_demo',
        strategy=strategy)
    '''
    :param log_interval: 进度日志打印间隔，默认每10个step打印一次
    :param eval_interval: 模型评估的间隔，默认每100个step评估一次验证集
    :param save_ckpt_interval: 模型保存间隔，请根据任务大小配置，默认只保存验证集效果最好的模型和训练结束的模型
    :param use_cuda: 是否使用GPU训练，默认为False
    :param checkpoint_dir: 模型checkpoint保存路径, 若用户没有指定，程序会自动生成
    :param num_epoch: finetune的轮数
    :param batch_size: 训练的批大小，如果使用GPU，请根据实际情况调整batch_size
    :param enable_memory_optim: 是否使用内存优化， 默认为True
    :param strategy: Finetune优化策略
    '''

    # Step4: 构建网络并创建序列标注迁移任务进行Finetune
    # Use "sequence_output" for token-level output.
    sequence_output = outputs["sequence_output"]

    # feed_list的Tensor顺序不可以调整
    feed_list = [
        inputs["input_ids"].name,
        inputs["position_ids"].name,
        inputs["segment_ids"].name,
        inputs["input_mask"].name,
    ]

    # Define a sequence labeling finetune task by PaddleHub's API
    seq_label_task = hub.SequenceLabelTask(
        data_reader=reader,
        feature=sequence_output,
        feed_list=feed_list,
        max_seq_len=10000,
        num_classes=dataSet.num_labels,
        config=config)

    seq_label_task.finetune_and_eval()

    # NOTE:
    # outputs["sequence_output"]返回了ERNIE/BERT模型输入单词的对应输出,可以用于单词的特征表达。
    # feed_list中的inputs参数指名了ERNIE/BERT中的输入tensor的顺序，与SequenceLabelReader返回的结果一致。
    # hub.SequenceLabelTask通过输入特征，迁移的类别数，可以生成适用于序列标注的迁移任务SequenceLabelTask
