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
                    if len(line_x) > config.max_len:
                        # 直接按最大长度切分
                        # print("len(line_x): ", len(line_x))
                        sub_word_list = get_sub_list(line_x, config.max_len - 5, config.sep_word)
                        sub_label_list = get_sub_list(line_y, config.max_len - 5, config.sep_label)
                        # x_train.extend(sub_word_list)
                        # y_train.extend(sub_label_list)
                    else:
                        x_train.append(line_x)
                        y_train.append(line_y)
                    line_x = []
                    line_y = []
                    continue
                line = line.split(' ')
                line_x.append(line[0])
                line_y.append(line[1]) # self.tag2id[line[1]]

        np.savez_compressed(output_dir, words=x_train, labels=y_train)
        print("length of x_train: ", len(x_train))
        print("length of y_train: ", len(y_train))
        print("x_train[0]: ", x_train[0])
        print("y_train[0]: ", y_train[0])
        for i in range(len(x_train)):
            if len(x_train[i]) > 512:
                print("i: ", i)
                print("len: ", len(x_train[i]))
        logging.info("-------- {} data process DONE!--------".format(mode))

def get_sub_list(init_list, sublist_len, sep_word):
    """直接按最大长度切分"""
    list_groups = zip(*(iter(init_list),) * sublist_len)
    end_list = [list(i) + list(sep_word) for i in list_groups]
    count = len(init_list) % sublist_len
    if count != 0:
        end_list.append(init_list[-count:])
    else:
        end_list[-1] = end_list[-1][:-1]  # remove the last sep word
    return end_list


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
