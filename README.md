# handwritten
a simple demo for handwritten by using python.


## 安装环境
``` cmd
pip install alive_progress
```

## 使用卷积训练模型
``` cmd
pip install alive_progress
export DATA_SET_FOLDER=/gemini/data-1
export MODEL_FOLDER=/gemini/pretrain
export TRAIN_FOLDER=PotTrain
export TEST_FOLDER=PotTest
cd /gemini/code/handwritten
python handwritten_training_cnn.py
```


## 使用Grad-12算法训练模型

数据预处理
``` cmd
export DATA_SET_FOLDER=work
export MODEL_FOLDER=pretrain
export TRAIN_FOLDER=PotTrain
export TEST_FOLDER=PotTest
export BATCH_SIZE=1024
export NUM_EPOCHS=200
export MIN_LN=0.0001
python dataset/pot_to_csv_grad_12.py
```


训练模型
``` cmd
export DATA_SET_FOLDER=work
export MODEL_FOLDER=pretrain
export TRAIN_FOLDER=PotTrain
export TEST_FOLDER=PotTest
export BATCH_SIZE=1024
export NUM_EPOCHS=200
export MIN_LN=0.0001
python handwritten_training_grad_12.py
```