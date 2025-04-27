PyTorch 文本分类项目
项目概述
这是一个基于PyTorch框架的中文文本分类系统，主要使用循环神经网络(RNN)对中文新闻文本进行分类。项目包含完整的训练、评估和部署流程。

主要功能
中文文本分类（基于THUCNews数据集）
支持多种深度学习模型（TextRNN, TextCNN等）
GPU加速支持（CUDA和MPS）
模型训练与评估
Flask服务部署
项目结构
pytorch/  
├── program/                  # 主程序目录  
│   ├── models/               # 模型定义  
│   │   └── TextRNN.py        # TextRNN模型  
│   ├── THUCNews/             # 数据集  
│   │   ├── data/             # 数据文件  
│   │   │   ├── train.txt     # 训练集  
│   │   │   ├── dev.txt       # 验证集  
│   │   │   ├── test.txt      # 测试集  
│   │   │   ├── class.txt     # 类别列表  
│   │   │   ├── vocab.pkl     # 词汇表  
│   │   │   └── embedding_SougouNews.npz  # 预训练词向量  
│   │   └── saved_dict/       # 保存的模型  
│   └── run.py                # 主运行脚本  
├── test/                     # 测试目录  
│   └── GPUtest.ipynb         # GPU性能测试  
└── flask/                    # Flask部署  
    ├── flask_server.py       # 服务器  
    └── flask_predict.py      # 预测客户端  
数据集
项目使用THUCNews数据集，这是一个中文新闻文本分类数据集，包含10个类别：

财经（finance）
房产（realty）
股票（stocks）
教育（education）
科技（science）
社会（society）
政治（politics）
体育（sports）
游戏（game）
娱乐（entertainment） dev.txt:391-395
模型
项目实现了多种深度学习模型，默认使用TextRNN：

TextRNN架构：  
输入文本 -> 词嵌入层 -> 双向LSTM -> 全连接层 -> 分类输出  
主要配置参数：

词嵌入维度：300
LSTM隐藏层大小：128
LSTM层数：2
Dropout率：0.5
批处理大小：129
学习率：1e-3 TextRNN.py:11-38
使用方法
环境要求
Python 3.6+
PyTorch 2.6.0+
TensorboardX
Flask (用于部署)
训练模型
# 使用默认TextRNN模型和搜狗新闻预训练词向量  
python run.py --model TextRNN  
  
# 使用随机初始化的词向量  
python run.py --model TextRNN --embedding random  
  
# 使用词级别分词（默认为字符级）  
python run.py --model TextRNN --word True
run.py:9-14

部署服务
# 启动Flask服务器  
cd flask  
python flask_server.py  
  
# 使用客户端进行预测  
python flask_predict.py --file 图片路径
flask_server.py:74-80

性能测试
项目包含GPU性能测试，展示了GPU加速相比CPU的显著优势：

操作	CPU时间	GPU时间	加速比
运行1	0.519秒	0.004秒	~129倍
运行2	0.130秒	0.00004秒	~3250倍

开发者
yw5579358 (最近提交: 2025-04-15, f32579dd)



