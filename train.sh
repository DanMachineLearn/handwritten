

# 修改训练参数
# 数据集位置
export DATA_SET_FOLDER=work
# 输出模型位置
export MODEL_FOLDER=pretrain
# 使用的训练集
export TRAIN_FOLDER=PotTrain
# 使用的测试集
export TEST_FOLDER=PotTest

# 其他超参数
export BATCH_SIZE=512
export NUM_EPOCHS=300

python handwritten_training_cnn.py