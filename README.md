# 手写识别项目
使用pytorch实现的简单单个手写汉字识别项目。


## 安装环境
``` cmd
pip install -r requirements.txt
```

## 下载数据集
``` cmd
python utils/pot_downloader.py
```

## 数据预处理（转jpg）
``` cmd
python dataset/handwritten_pot_dataset.py
```

## 训练模型
``` cmd
export DATA_SET_FOLDER=work
export MODEL_FOLDER=pretrain
export TRAIN_FOLDER=PotTrain
export TEST_FOLDER=PotTest
export NUM_EPOCHS=11
export BATCH_SIZE=512
python handwritten_training_googlenet.py
```


## 随便说点

这是我的第一个神经网络训练项目。整个模型是基于GoogLeNet训练的，能识别3000多个一级汉字 + 单个英文字母 + 标点符号（主要是因为 中科院的数据集就这么多），验证集的准确率达到94%。当然源码里面还包含了其他算法，我也懒得去精简代码了，就这样吧。