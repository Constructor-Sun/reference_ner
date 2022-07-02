import os
import re
import config
import logging
import numpy as np
import pickle


def getlist(input_str):
    """
    将每个输入词转换为BMES标注
    """
    output_str = []
    if len(input_str) == 1:
        output_str.append('S')
    elif len(input_str) == 2:
        output_str = ['B', 'E']
    else:
        M_num = len(input_str) - 2
        M_list = ['M'] * M_num
        output_str.append('B')
        output_str.extend(M_list)
        output_str.append('E')
    return output_str


class Processor:
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.files = config.files
        self.id2tag = []
        self.tag2id = {}

    def process(self):
        for file_name in self.files:
            self.get_examples(file_name)

    
    def get_examples(self, mode):
        # """
        # 将txt文件每一行中的文本分离出来，存储为words列表
        # BMES标注法标记文本对应的标签，存储为labels
        # 若长度超过max_len，则直接按最大长度切分（可替换为按标点切分）
        # """
        input_dir = self.data_dir + str(mode) + '.txt'
        output_dir = self.data_dir + str(mode) + '.npz'
        if os.path.exists(output_dir) is True:
            return
        unique = set()
        with open(input_dir, 'r')as f:
            for line in f:
                try:
                    unique.update([line.strip('\n').split(' ')[1]])
                except:
                    pass
        self.id2tag = sorted(unique) # in fact, it is the unique values of labels
        self.id2tag = sorted(unique) # in fact, it is the unique values of labels
        for i, label in enumerate(self.id2tag):
            self.tag2id[label] = i

        x_train = []
        y_train = []

        with open(input_dir, 'r', encoding="utf-8") as ifp:
            line_x = []
            line_y = []
            for line in ifp:
                line = line.strip()
                if not line: # meaning the end of the training_data file?
                    x_train.append(line_x)
                    y_train.append(line_y)
                    line_x = []
                    line_y = []
                    continue
                line = line.split(' ')
                line_x.append(line[0])
                line_y.append(self.tag2id[line[1]])

        np.savez_compressed(output_dir, words=x_train, labels=y_train)
        print("length of x_train: ", len(x_train))
        print("length of y_train: ", len(y_train))
        print("x_train[0]: ", x_train[0])
        print("y_train[0]: ", y_train[0])
        logging.info("-------- {} data process DONE!--------".format(mode))


        """with open(input_dir, 'r', encoding='utf-8') as f:
            word_list = []
            label_list = []
            num = 0
            sep_num = 0
            for line in f:
                words = []
                line = line.strip()  # remove spaces at the beginning and the end
                if not line:
                    continue  # line is None
                for i in range(len(line)):
                    if line[i] == " ":
                        continue  # skip space
                    words.append(line[i])
                text = line.split(" ")
                labels = []
                for item in text:
                    if item == "":
                        continue
                    labels.extend(getlist(item))
                if len(words) > config.max_len:
                    # 直接按最大长度切分
                    sub_word_list = get_sub_list(words, config.max_len - 5, config.sep_word)
                    sub_label_list = get_sub_list(labels, config.max_len - 5, config.sep_label)
                    word_list.extend(sub_word_list)
                    label_list.extend(sub_label_list)
                    sep_num += 1
                else:
                    word_list.append(words)
                    label_list.append(labels)
                num += 1
                assert len(labels) == len(words), "labels 数量与 words 不匹配"
                assert len(word_list) == len(label_list), "label 句子数量与 word 句子数量不匹配"
            print("We have", num, "lines in", mode, "file processed")
            print("We have", sep_num, "lines in", mode, "file get sep processed")
            # 保存成二进制文件
            np.savez_compressed(output_dir, words=word_list, labels=label_list)
            logging.info("-------- {} data process DONE!--------".format(mode))"""


def get_process():
    if os.path.exists(config.train_dir):
        os.remove(config.train_dir)
    if os.path.exists(config.test_dir):
        os.remove(config.test_dir)
    # 处理数据，分离文本和标签
    processor = Processor(config)
    processor.process()



if __name__ == "__main__":
    get_process()
    # print_len()
