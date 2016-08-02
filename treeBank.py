#!/usr/bin/python
#coding:utf-8
__author__ = 'Tuxun tuxun1988@gmail.com'


import sys
import re
import os
import numpy as np

class HmmMatrix(object):
    '''
    使用Chinese tree bank 8.0标注数据集训练hmm的转移矩阵和混淆矩阵,
    并生成词汇表等工具
    '''

    _SPLIT = " " 

    def __init__(self):
        '''
        初始化
        '''
        self.wordTagList = [] #词汇,标注二元组列表
        self.tagTagList = []  #标注,标注二元组列表
        self.wordVocab = {}   #词汇,序号对应表
        self.tagVocab = {}    #标注,序号对应表
        self.wordTotal = {}   #词汇出现次数统计
        self.tagTotal = {}    #标注出现次数统计
        self.reTagVocab = {}  #序号,词汇反查表


    def reverseTagVocab(self):
        '''
        构建序号,词汇反查表
        '''
        for i in self.tagVocab:
            self.reTagVocab[self.tagVocab[i]] = i


    def _tokenizer(self, line):
        '''
        句子分割成词,标注二元组,并存入两个list里
        '''
        #按照正则分割句子
        wordlist= re.split(HmmMatrix._SPLIT, line)
        lastTag = None
        for word in wordlist:
            if '_' in word:
                #按照_分割成二元组
                pword = tuple(word.split('_'))
                if len(pword) == 2:
                    #更新词,标注列表
                    self.wordTagList.append(pword)
                    if lastTag:
                        #更新标注,标注列表
                        self.tagTagList.append((lastTag, pword[1]))
                    else:
                        pass
                    lastTag = pword[1]
                else:
                    pass
            else:
                pass


    def _lineFilter(self, line):
        '''
        从原始文本中过滤出所需的句子
        '''
        line = line.strip()
        if len(line) <= 1:
            pass
        elif line[0] == '<' and line[-1] == '>':
            pass
        else:
            self._tokenizer(line)


    def pathCount(self, path):
        '''
        处理路径下所有文件
        '''
        filelist = os.listdir(path)
        for f in filelist:
            fp = os.path.join(path, f)
            with open(fp) as fr:
                for line in fr:
                    self._lineFilter(line)

    
    def createVocab(self):
        '''
        建立映射字典
        '''
        wordnum = 0
        tagnum = 0
        for pair in self.wordTagList:
            word = pair[0]
            tag = pair[1]
            if word not in self.wordVocab:
                self.wordVocab[word] = wordnum
                self.wordTotal[wordnum] = 1
                wordnum += 1
            else:
                self.wordTotal[self.wordVocab[word]] += 1
            if tag not in self.tagVocab:
                self.tagVocab[tag] = tagnum
                self.tagTotal[tagnum] = 1
                tagnum += 1
            else:
                self.tagTotal[self.tagVocab[tag]] += 1


    def createTransformMatrix(self):
        '''
        创建状态转移矩阵
        '''
        m = len(self.tagVocab)
        transformMatrix = np.zeros((m, m))
        for pair in self.tagTagList:
            pos1 = self.tagVocab[pair[0]]
            pos2 = self.tagVocab[pair[1]]
            transformMatrix[pos1, pos2] += 1.0
        for i in range(0, m):
            #这里直接除以行之和,因为tagTotal字典里统计了句子结尾的词,但
            #转移矩阵的分子统计不到
            transformMatrix[i, :] = transformMatrix[i, :] / np.sum(transformMatrix[i, :])
        return transformMatrix


    def createConfusionMatrix(self):
        '''
        创建混淆矩阵
        '''
        m = len(self.tagVocab)
        n = len(self.wordVocab)
        confusionMatrix = np.zeros((m, n))
        for pair in self.wordTagList:
            word = self.wordVocab[pair[0]]
            tag = self.tagVocab[pair[1]]
            confusionMatrix[tag, word] += 1.0
        for i in range(0, m):
            confusionMatrix[i, :] = confusionMatrix[i, :] / float(self.tagTotal[i])
        return confusionMatrix


if __name__ == '__main__':
    H = HmmMatrix()
    H.pathCount('dictionary/ctb8.0/data/postagged/')
    H.createVocab()
    A = H.createTransformMatrix()
