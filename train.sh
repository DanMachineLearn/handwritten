


# 启动训练窗口
# tmux new -s handwritten
# 返回上一次训练
# tmux attach -t handwritten

# 修改训练参数
# 数据集位置
export DATA_SET_FOLDER=work
export MODEL_FOLDER=pretrain
export TRAIN_FOLDER=PotTrain
export TEST_FOLDER=PotTest
export BATCH_SIZE=1024
export NUM_EPOCHS=200
export MIN_LN=0.0000001

python handwritten_training_cnn.py