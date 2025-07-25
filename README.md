### 任务一：基于机器学习的文本分类

实现基于logistic/softmax regression的文本分类

1. 参考
   1. [文本分类](文本分类.md)
   2. 《[神经网络与深度学习](https://nndl.github.io/)》 第2/3章
2. 数据集：[Classify the sentiment of sentences from the Rotten Tomatoes dataset](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)
3. 实现要求：NumPy
4. 实验：
   1. 分析不同的特征、损失函数、学习率对最终分类性能的影响
   2. shuffle 、batch、mini-batch
<img width="1200" height="500" alt="training_history1" src="https://github.com/user-attachments/assets/039f298c-b132-496f-986a-ee7505c96f1d" />

### 任务二：基于深度学习的文本分类

熟悉Pytorch，用Pytorch重写《任务一》，实现CNN、RNN的文本分类；

1. 参考

   1. https://pytorch.org/
   2. Convolutional Neural Networks for Sentence Classification <https://arxiv.org/abs/1408.5882>
   3. <https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/>
2. word embedding 的方式初始化
3. 实验
<img width="1200" height="500" alt="training_history_CNN_20250725_114904" src="https://github.com/user-attachments/assets/ddc6f245-67ae-4ad4-aab1-ec0d552c0519" />

### 任务三：基于注意力机制的文本匹配

输入两个句子判断，判断它们之间的关系。参考[ESIM]( https://arxiv.org/pdf/1609.06038v3.pdf)（可以只用LSTM，忽略Tree-LSTM），用双向的注意力机制实现。

1. 参考
   1. 《[神经网络与深度学习](https://nndl.github.io/)》 第7章
   2. Reasoning about Entailment with Neural Attention <https://arxiv.org/pdf/1509.06664v1.pdf>
   3. Enhanced LSTM for Natural Language Inference <https://arxiv.org/pdf/1609.06038v3.pdf>
2. 数据集：https://nlp.stanford.edu/projects/snli/
3. 实现要求：Pytorch
4. 实验
<img width="1200" height="500" alt="esim_training_history" src="https://github.com/user-attachments/assets/fc68a894-09ba-4ffa-b315-f6b1c1021a2f" />
生成样例：
Premise: This church choir sings to the masses as they sing joyous songs from the book at a church.
Hypothesis: The church has cracks in the ceiling.
True Label: neutral
Predicted Label: contradiction
--------------------------------------------------
Premise: This church choir sings to the masses as they sing joyous songs from the book at a church.
Hypothesis: The church is filled with song.
True Label: entailment
Predicted Label: neutral

### 任务四：基于LSTM+CRF的序列标注

用LSTM+CRF来训练序列标注模型：以Named Entity Recognition为例。

1. 参考
   1. 《[神经网络与深度学习](https://nndl.github.io/)》 第6、11章
   2. https://arxiv.org/pdf/1603.01354.pdf
   3. https://arxiv.org/pdf/1603.01360.pdf
2. 数据集：CONLL 2003，https://www.clips.uantwerpen.be/conll2003/ner/
3. 实现要求：Pytorch
4. 实验
5. <img width="1200" height="500" alt="lstm_training_history" src="https://github.com/user-attachments/assets/07c5df4a-6b06-4dde-af58-b377d3616ef4" />
[lstm_generated_text.txt](https://github.com/user-attachments/files/21433340/lstm_generated_text.txt)
生成样例：
月寒，
不见禁群十峻红。已君不复发，虽月翠老幽。
卿君多国白，风莫溪携人。
今愧颇求断，薄我不得涯。莫但徒子狭，开鸟烛绿名。
君来与子合，犹枚已啄。竹柢下有下，觉是南不心。

子盱翻嶂冢，委我来知。维为闻按落，鱼薄故侯深。
当料芳琐，动入移台。
枯蕊万入国，威公在可拭。竹客会智，不肥人。

天上延者，重谷多如。何柢已，江江期。岂知别头旧，遗流逐波艇。

惜山千凫，咿敌汉。以鸟喜河，虔鸟定。威之斗，降以斫。鬼测兕，连罗舒。
所僚汲，九而华。系，深子弥，浮。百露烈，此寂巧。
放出旧，因吴毛。黎鸟德，抗仑讨。浩豺弥，一骑天。
官人不观，虚原理人。 

栖攘亡入，涨可彩。乏驴莫，取海雹。王退北逡突，天巷
### 任务五：基于神经网络的语言模型

用LSTM、GRU来训练字符级的语言模型，计算困惑度

1. 参考
   1. 《[神经网络与深度学习](https://nndl.github.io/)》 第6、15章
2. 数据集：poetryFromTang.txt
3. 实现要求：Pytorch
4. 实验
   <img width="3600" height="1500" alt="training_history" src="https://github.com/user-attachments/assets/e54e6e6d-ccc5-4840-827a-8eddb7beabb4" />
<img width="3600" height="1800" alt="entity_metrics" src="https://github.com/user-attachments/assets/dca1a44c-ca44-4bab-a87c-fb679fbecf7a" />

