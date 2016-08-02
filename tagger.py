#!/usr/bin/python
#coding:utf-8
__author__ == 'Tuxun tuxun1988@gmail.com'

import numpy as np
import treeBank
import sys


class POStagging(object):
    '''
    POS tagging
    '''
    
    def __init__(self):
        self.H = treeBank.HmmMatrix()
        self.H.pathCount('dictionary/ctb8.0/data/postagged/')
        self.H.createVocab()     #创建映射表
        self.H.reverseTagVocab() #创建反查表
        self.A = self.H.createTransformMatrix()  #创建状态转移矩阵
        self.B = self.H.createConfusionMatrix()  #创建混淆矩阵
        self.m = len(self.H.tagVocab)     #标注种类数目
        self.n = len(self.H.wordVocab)    #词汇种类数目
        pi = np.random.randn((self.m))    #初始状态矩阵
        self.pi = pi / np.sum(pi)         #归一化
        

    def prep(self, obseq):
        '''
        利用映射表把词映射成序号
        '''
        outlist = []
        for i in obseq:
            if i in self.H.wordVocab:
                outlist.append(self.H.wordVocab[i])
        return outlist


    def viterbi(self, obseq):
        '''
        维特比算法
        '''
        seq = self.prep(obseq)   #映射后的观察序列
        t = len(seq)             #序列长度
        z = np.zeros((t, self.m))  #最大概率矩阵
        d = np.zeros((t, self.m), dtype=int)  #前一个状态矩阵

        #序列为0的时候的z,d矩阵
        for i in range(self.m):
            z[0, i] = self.pi[i] * self.B[i, seq[0]]
            d[0, i] = 0

        #按照序列前进方向确定z,d矩阵
        for j in range(1, t):
            for i in range(self.m):
                maxN = z[j-1, 0] * self.A[0, i]
                mark = 0
                for k in range(1, self.m):
                    p = z[j-1, k] * self.A[k, i]
                    if p > maxN:
                        maxN = p
                        mark = k
                z[j, i] = maxN * self.B[i, seq[j]]
                d[j, i] = mark
                
        #隐藏状态序列,从后往前推算
        tagseq = np.zeros((t), dtype=int)
        tagseq[t-1] = z[t-1,:].argmax()
        for s in range(t-2, -1, -1):
            tagseq[s] = d[s+1, tagseq[s+1]]
        tagseq = list(tagseq)

        #把隐藏序列转化为标注
        tagNameSeq = []
        for s1 in tagseq:
            tagNameSeq.append(self.H.reTagVocab[s1])
        outline = ['_'.join(pair) for pair in zip(obseq, tagNameSeq)]

        #输出词,词性对列表
        return outline
        

if __name__ ==  '__main__':
    testline = ['我','爱','北京','天安门','。']
    POS = POStagging()
    print ' '.join(POS.viterbi(testline))
