# Readme
## google drive link
https://drive.google.com/drive/folders/1NwBbIW98TXI5ILyVfZbzu3m4GOTc_drA?usp=sharing

## original source
Here are original source for my code base

https://github.com/sriniiyer/concode

https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/text-to-code


## list of files that I modified
In the concode-master folder, I reproduce the RegularEncoder.py and RegularDecoder.py and modify the architecture of the model. Also, I modify the UtilClass.py and S2Smodel.py to make sure the code can successfully be run with my modification. I also tried to produce a GRUEncoder.py and GRUDecoder.py. In the ConcodeEncoder.py and ConcodeDecoder.py files, I tried different information to be added to reach the best performance.

In the pre-trained folder, I fine-tune the model in the run.py in the main function. I smooth the pre-trained model and tried to get a better performance.

## A list of the major software requirements
### concode master environment

antlr4-python3-runtime==4.6

allennlp==0.3.0

ipython==6.5

torch==0.3.0

torchvision==0.2.0

### how to run the code 
download data
```
mkdir concode
cd concode
```
Download data from: https://drive.google.com/drive/folders/1kC6fe7JgOmEHhVFaXjzOmKeatTJy1I1W into this folder.

build dataset
```
!python build.py -train_file concode/train_shuffled_with_path_and_id_concode.json -valid_file concode/valid_shuffled_with_path_and_id_concode.json -test_file concode/test_shuffled_with_path_and_id_concode.json -output_folder data -train_num 100000 -valid_num 2000
```
preprocess data

```
mkdir data/d_100k_762
```
```
!python preprocess.py -train data/train.dataset -valid data/valid.dataset -save_data data/d_100k_762/concode -train_max 50000 -valid_max 2000
```
train model
```
!python train.py -dropout 0.5 -data data/d_100k_762/concode -save_model data/d_100k_762/s2s -epochs 30 -learning_rate 0.001 -seed 1123 -enc_layers 2 -dec_layers 2 -batch_size 20 -src_word_vec_size 1024 -tgt_word_vec_size 512 -rnn_size 1024 -encoder_type regular -decoder_type regular -copy_attn
```
test model
```
!ipython predict.ipy -- -start 1 -end 3 -beam 3 -models_dir data/d_100k_762/concode -test_file data/valid.dataset -tgt_len 500
```

### pretrained model environment

torch==1.4.0

torchvision==0.5.0

transformers-4.0.

### how to run the code 
train data
```
!python -m torch.distributed.launch --nproc_per_node=1 run.py \
        --data_dir=../dataset/concode \
        --langs=java \
        --output_dir=../save/concode \
        --pretrain_dir=microsoft/CodeGPT-small-java-adaptedGPT2 \
        --log_file=text2code_concode.log \
        --model_type=gpt2 \
        --block_size=512 \
        --do_train \
        --node_index 0 \
        --gpu_per_node 1 \
        --learning_rate=5e-5 \
        --weight_decay=0.01 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=3 \
        --per_gpu_eval_batch_size=6 \
        --gradient_accumulation_steps=2 \
        --num_train_epochs=30 \
        --logging_steps=100 \
        --save_steps=1000 \
        --overwrite_output_dir \
        --seed=42
```

evaluate model 
```
!python -u run.py \
        --data_dir=../dataset/concode \
        --langs=java \
        --output_dir=../save/concode \
        --pretrain_dir=../save/concode/checkpoint-5000-16.91 \
        --log_file=text2code_concode_eval.log \
        --model_type=gpt2 \
        --block_size=512 \
        --do_eval \
        --logging_steps=100 \
        --seed=42
```