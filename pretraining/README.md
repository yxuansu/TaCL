### Code for Training CLBERT

**[Note]** Before training CLBERT, please make sure you have prepared the pre-training corpora as described [here](https://github.com/yxuansu/CLBERT/tree/main/pretraining_data).

To debug whether you have everything ready, you can run test scripts as

```yaml
chmod +x ./debug_clbert_{}.sh
./debug_clbert_{}.sh
```
Here, {} is in ['english', 'chinese'].

and some key parameters are described below:

```yaml
--use_nlu: Whether to include pre-training data that is annotated for NLU task. The default value is True.

--use_bs: Whether to include pre-training data that is annotated for DST task. The default value is True.

--use_da: Whether to include pre-training data that is annotated for POL task. The default value is True.

--use_nlg: Whether to include pre-training data that is annotated for NLG task. The default value is True.

--gradient_accumulation_steps: How many forward computations between two gradient updates.

--number_of_gpu: Number of avaliable GPUs.

--batch_size_per_gpu: The batch size for each GPU.
```

**[Note]** The actual batch size equals to gradient_accumulation_steps x number_of_gpu x batch_size_per_gpu. We recommend
you to set the actual batch size value as 128. All PPTOD models are trained on a single machine with 8 x Nvidia V100 GPUs (8 x 32GB memory).
