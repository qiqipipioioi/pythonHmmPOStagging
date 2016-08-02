#hmm算法POS tagging的python实现

##算法说明
利用Chinese tree bank 8.0词性标注数据集作为训练集，建立混淆矩阵和状态转移矩阵，然后用viterbi算法对新的句子实现自动词性标注。

##使用说明
先实例化POStagging类，然后调用viterbi方法生成输出列表  
例如  
```
testline = ['我','爱','北京','天安门','。']
POS = POStagging()
print ' '.join(POS.viterbi(testline))

我_PN 爱_VV 北京_NR 天安门_NR 。_PU
```

##脚本
```
dictionary/     #tree bank字典
tagger.py       #标注类
treeBank.py     #生成hmm矩阵类
```

##tagSet
出处详见http://repository.upenn.edu/cgi/viewcontent.cgi?article=1039&context=ircs_reports

(https://github.com/qiqipipioioi/pythonHmmPOStagging/blob/master/tagging.JPG)
