cd /home/hjz/work/Classifier

TASK=L
INPUT_IMG=CT_front
IN_SIZE=512,410
LOSS_FUN=FocalLoss
# OPTIMIZER=Adam
# OPTIMIZER=SGD
OPTIMIZER=RMSprop
BATCH_SIZE=8

# TASK=T
# ALL_LABEL=1-10,11-20,21-30,31-40,41-50

# TASK=L
# ALL_LABEL=1-30,31-40,41-50,51-60,60+

# TASK=TL
# ALL_LABEL=1-20,21-30,31-40,41+

# TASK=Type
# ALL_LABEL=NA,C,S

TASK=Cobb
ALL_LABEL=0-5,6-10,11-15,16-20,21-25


DATA_DIR=/home/hjz/data/scoliosis/clean_new/
# TASK_NAME=back_new_resized_KFold
TASK_NAME=CT_front_Sample_Task_NoPretrain_NoWeight
# MODE_NAME=AlexNet # bs 8
MODE_NAME=ResNet50 # bs 8
# MODE_NAME=swintransformer # bs 2
# MODE_NAME=SKNet #bs 4
# MODE_NAME=AlexNet #bs 16
# MODE_NAME=densenet #bs 2
#MODE_NAME=vgg #bs 8
#MODE_NAME=Inception # bs 4 no
# MODE_NAME=SKNet # bs 4

# CUDA_VISIBLE_DEVICES=$1 nohup /mnt/data/hejz/anaconda3/envs/JD/bin/python -u train.py --task_name $TASK_NAME --root $DATA_DIR --batch_size 8 --gpu_num $2 --lr 0.006 --num_classes 5 --model_name $MODE_NAME --loss_func FocalLoss >> train_$TASK_NAME$MODE_NAME.out &
#CUDA_VISIBLE_DEVICES=$1 nohup /home/hjz/.conda/envs/torch/bin/python -u train_KFold.py --task_name $TASK_NAME --root $DATA_DIR --batch_size 4 --gpu_num $2 --in_size 1000,680 --input_img back_resized  --seed 3107 --lr 0.06 --epochs 200 --num_classes 5 --model_name $MODE_NAME --loss_func BCE >> train_$TASK_NAME$MODE_NAME.out &
CUDA_VISIBLE_DEVICES=$1 nohup /home/hjz/.conda/envs/torch/bin/python -u train_SampTask.py --task_name $TASK_NAME --root $DATA_DIR --batch_size $BATCH_SIZE --gpu_num $2 --in_size $IN_SIZE --input_img $INPUT_IMG  --seed 3407 --lr 0.06 --epochs 200 --task $TASK --train_label $ALL_LABEL --model_name $MODE_NAME --loss_func $LOSS_FUN --opt $OPTIMIZER >> log_train_$TASK_NAME\_$TASK\_$MODE_NAME\_$OPTIMIZER.out &
