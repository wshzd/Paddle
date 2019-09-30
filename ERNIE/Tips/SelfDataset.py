from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np
import pandas as pd
from paddlehub.dataset import InputExample, HubDataset
#InputExample 上面的代码的主要的目的就是把数据拼到这个类里面
#这个部分是为了保证模型数据集在随机条件下展示出更精准的模型能力
def random(data):
    random_order = list(range(len(data)))
    np.random.shuffle(random_order)
    #模型数据分配是 训练集 ：验证集：测试集 8:1:1
    train_data = [data[j] for i, j in enumerate(random_order) if i % 3 == 0]
    valid_data = [data[j] for i, j in enumerate(random_order) if i % 24 == 1]
    test_data = [data[j] for i, j in enumerate(random_order) if i % 24 == 2]
    return train_data, valid_data, test_data


def read_message():
    if not os.path.exists("sets/without_data.pkl"):
        x_items = []
        #获取用户信息
        user_message = pd.read_csv("/home/aistudio/data/data10296/table1_user",
                                   sep="\t")
        #获取岗位信息
        jd_message = pd.read_csv("/home/aistudio/data/data10296/table2_jd",
                                 sep="\t")
        #获取岗位曝光数据 以及场景数据
        match_message = pd.read_csv("/home/aistudio/data/data10296/table3_action",
                                    sep="\t")
        user_message_index = {}
        for i in user_message.values.tolist():
            user_message_str = ''
            for message in i[1:]:
                user_message_str += str(message)
            user_message_index[i[0]] = user_message_str
        jd_message_index = {}
        for i in jd_message.values.tolist():
            user_message_str = ''
            for message in i[1:]:
                user_message_str += str(message)
            jd_message_index[i[0]] = user_message_str
        for i in match_message.values.tolist():
            if i[0] in user_message_index.keys():
                x_item = str(user_message_index[i[0]])
            else:
                continue
            if i[1] in jd_message_index.keys():
                x_item += str(jd_message_index[i[1]])
            else:
                continue
            y_label = str(i[2]) + str(i[3]) + str(i[4])
            if y_label != '000':
                c = [x_item, y_label]
                x_items.append(c)
        with open('sets/without_data.pkl', 'wb') as f:
            pickle.dump(x_items, f)
    else:
        with open('sets/without_data.pkl', 'rb') as f:
            x_items = pickle.load(f)
    train_data, valid_data, test_data = random(x_items)
    return train_data, valid_data, test_data


def _read_tsv(input_file):
    """Reads a tab separated value file."""
    examples = []
    seq_id = 0
    for line in input_file:
        #这一步非常的中药就是我们拼接 text_a 使用未分割的字符串（框架内部有实现分词）
        #label这个参数拼接分类类别
        example = InputExample(
            guid=seq_id, label=line[1], text_a=line[0])
        seq_id += 1
        examples.append(example)
    return examples


class DemoDataset(HubDataset):
    """DemoDataset"""

    def __init__(self):
        self.dataset_dir = "path/to/dataset"
        self.train_data, self.valid_data, self.test_data = read_message()
        self._load_train_examples()
        self._load_test_examples()
        self._load_dev_examples()

    def _load_train_examples(self):
    """载入训练数据"""
        self.train_examples = _read_tsv(self.train_data)

    def _load_dev_examples(self):
    """载入验证集数据"""
        self.dev_examples = _read_tsv(self.valid_data)

    def _load_test_examples(self):
    """载入测试集数据"""
        self.test_examples = _read_tsv(self.test_data)

    def get_train_examples(self):
        return self.train_examples

    def get_dev_examples(self):
        return self.dev_examples

    def get_test_examples(self):
        return self.test_examples

    def get_labels(self):
        """define it according the real dataset
        设置分类标签（这里可以在read_message中增加一个返回参数，但是我并没有加
        如果替换数据集记得修改此处代码）
        """
        
        return ["000", "100", "110", "111","010"]

    @property
    def num_labels(self):
        """
        Return the number of labels in the dataset.
        """
        return len(self.get_labels())



import paddlehub as hub
module = hub.Module(name="ernie", version="1.0.2")
dataset = DemoDataset()


reader = hub.reader.ClassifyReader(
    dataset=dataset,
    vocab_path=module.get_vocab_path(),
    max_seq_len=128)

strategy = hub.AdamWeightDecayStrategy(
    weight_decay=0.01,
    warmup_proportion=0.1,
    learning_rate=1e-5,
    lr_scheduler="linear_decay",
    optimizer_name="adam")

config = hub.RunConfig(
    #是否使用GPU
    use_cuda=True,
    num_epoch=50,
    #模型保存地址
    checkpoint_dir="ernie_turtorial_demo",
    batch_size=64,
    log_interval=10,
    eval_interval=500,
    strategy=strategy)
inputs, outputs, program = module.context(
    trainable=True, max_seq_len=128)

#对整个句子中的分类任务使用“pooled_output”。Use "pooled_output" for classification tasks on an entire sentence.
pooled_output = outputs["pooled_output"]

feed_list = [
    inputs["input_ids"].name,
    inputs["position_ids"].name,
    inputs["segment_ids"].name,
    inputs["input_mask"].name,
]
#配置模型参数 
cls_task = hub.TextClassifierTask(
    data_reader=reader,
    feature=pooled_output,
    
    feed_list=feed_list,
    #分类数量 在自定义的数据集的时候可以进行裁剪机
    num_classes=dataset.num_labels,
    config=config)
#开始训练模型
cls_task.finetune_and_eval()
