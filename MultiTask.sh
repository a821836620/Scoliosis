cd /home/hjz/work/Classifier

INPUT_IMG=CT_front
IN_SIZE=512,512
LOSS_FUN=FocalLoss
OPTIMIZER=Adam

TASK_NAME=MultiTask_CT_front_lr0.0001_new
MODE_NAME=ResNet50
BATCH_SIZE=16
L_R=0.0001

DATA_DIR=/home/hjz/data/scoliosis/clean_new/


CUDA_VISIBLE_DEVICES=$1 nohup /home/hjz/.conda/envs/torch/bin/python -u train_MultiTask.py --task_name $TASK_NAME --root $DATA_DIR --batch_size $BATCH_SIZE --gpu_num $2 --in_size $IN_SIZE --input_img $INPUT_IMG  --seed 3407 --lr $L_R --epochs 500  --loss_func $LOSS_FUN --opt $OPTIMIZER >> log_train_$TASK_NAME\_$MODE_NAME.out &
