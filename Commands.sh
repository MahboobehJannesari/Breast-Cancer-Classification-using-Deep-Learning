
#=======================convert image to jpg ==============================

find . -type f ! -name '*.jpg' -exec \
mogrify -format jpg -quality 100 {} + -exec rm {} +


#=======================catch_error_images==================================
#for deleting damaged images

cd ./0.Convert_to_jpg_Catch_Error

python3 catch_error_images.py 

#=======================converting to TFRecords============================
#tensorflow 1.5

cd ../1.Convert_to_TFrecord
python3.5 convert_to_TFrecord.py 

#======================= Training_by_inception_v3_3000_epochs ==============


# This script performs the following operations:
# 1. Fine-tunes an InceptionV3 model on the Breast Cancer (Malignant) training set.
# 2. Evaluates the model on the Breast Cancer (Malignant) validation set.
#
# Usage:
# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=../checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=../inception_v3

# Where the dataset used and TFRecords is saved to.
DATASET_DIR=../Project_Data
            

# Fine-tune only the new layers( Last Layers)  for 3000 steps. / TEST prog!!!
cd ../2.Classification_Training
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=cancers \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=3000 \
  --batch_size=4 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004


# Run Evaluation
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=cancers \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3


# Fine-tune all the new layers for 3000 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=cancers \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --checkpoint_path=${TRAIN_DIR} \
  --max_number_of_steps=3000\
  --batch_size=4 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=cancers \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3



#============================= Predcition_by_inception_v3=========================

cd ../3.Classification_Prediction

python3 Prediction.py --Test_dir=../Test_images --Batch_Size=1







