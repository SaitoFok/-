基于llama3的微调模型，可准确预测文本难度。需配合meta-llama-3___1-8b-instruct使用，sft还差adapter_model.safetensors未上传，需联系作者。

---
base_model: /root/autodl-tmp/llm-research/meta-llama-3___1-8b-instruct
library_name: peft
license: other
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: sft
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft

This model is a fine-tuned version of [/root/autodl-tmp/llm-research/meta-llama-3___1-8b-instruct](https://huggingface.co//root/autodl-tmp/llm-research/meta-llama-3___1-8b-instruct) on the abc dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1884

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 4
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 10.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.0819        | 3.1397 | 500  | 0.1991          |
| 0.0079        | 6.2794 | 1000 | 0.1342          |
| 0.0           | 9.4192 | 1500 | 0.1852          |


### Framework versions

- PEFT 0.12.0
- Transformers 4.45.0
- Pytorch 2.1.2+cu118
- Datasets 2.21.0
- Tokenizers 0.20.0
