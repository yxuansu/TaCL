## Code for Training CLBERT

### 1. Prepare data:
**[Note]** Before training CLBERT, please make sure you have prepared the pre-training corpora as described [here](https://github.com/yxuansu/CLBERT/tree/main/pretraining_data).

### 2. Make sure everything is ready:
To debug whether you have everything ready, you can run test scripts as

```yaml
chmod +x ./debug_clbert_{}.sh
./debug_clbert_{}.sh
```
Here, {} is in ['english', 'chinese'].

### 3. Train the mode:
After completing the test, you can train CLBERT as 
```yaml
chmod +x ./train_clbert_{}.sh
./train_clbert_{}.sh
```
Here, {} is in ['english', 'chinese'] and some key parameters are described below:

```yaml
--language: Which language you are training on. Should set to 'chinese' or 'english'
--model_name: The initial BERT model. For Chinese CLbert use 'bert-base-chinese', and for English CLBERT use 'bert-base-uncased'.
--train_data: The path stores your pre-training data.
--max_len: Maximum length of each sequence.
--number_of_gpu: Number of GPUs used to train the mode.
--batch_size_per_gpu: The batch size for each GPU.
--gradient_accumulation_steps: How many forward computations between two gradient updates.
--effective_batch_size: The overall batch size. It equals to batch_size_per_gpu x gradient_accumulation_steps x number_of_gpu.
--total_steps: Total training steps. In our paper, we train CLBERT for 150k steps.
--print_every: Have many steps to show the intermediate results.
--save_every: How many steps to save one checkpoint.
--ckpt_save_path: Where to save the checkpoints.
```

CLBERT can be trained on a single machine with 8 Nvidia V100 GPUs. For machines with different memory, the training configurations are listed as below.

|Memory per GPU|number_of_gpu|batch_size_per_gpu|gradient_accumulation_steps|effective_batch_size|
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|16GB|8|16|2|256|
|32GB|8|32|1|256|

### 4. Whole word masking option:


## Acknowledgements:


**[Note]** The actual batch size equals to gradient_accumulation_steps x number_of_gpu x batch_size_per_gpu. We recommend
you to set the actual batch size value as 128. All PPTOD models are trained on a single machine with 8 x Nvidia V100 GPUs (8 x 32GB memory).
