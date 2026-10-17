现在，让我们加载数据集和标记符：
```python
# 加载 Bespoke-Stratos-17k 数据集
# 为了简化，这里仅使用训练集划分
# load_dataset 函数从 Hugging Face Hub 加载指定的数据集
# split='train' 表示只加载训练集部分
dataset_sft = load_dataset("HuggingFaceH4/Bespoke-Stratos-17k", split='train')

# 初始化分词器 - 与之前相同
# 从预训练模型中加载分词器
# MODEL_NAME 是之前定义的模型名称，指定了要使用的模型
# trust_remote_code=True 表示信任远程代码，用于处理一些自定义代码
# padding_side="right" 表示在分词时从右侧进行填充
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="right"
)
# 如果分词器没有设置填充标记（pad_token）
# 则将其设置为结束标记（eos_token）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

最后，我们初始化 SFTTrainer 并开始训练！
```python
# 初始化用于监督微调（SFT）的基础模型 - 与之前相同
# 从预训练模型中加载基于因果语言模型的模型
# MODEL_NAME 为之前定义的模型名称，指定要使用的模型
# trust_remote_code=True 表示信任远程代码，用于处理自定义代码
# torch_dtype=torch.bfloat16 表示使用 bfloat16 数据类型，可减少内存占用和加速计算
model_sft = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# 初始化监督微调（SFT）训练器
sft_trainer = SFTTrainer(
    # 传入已初始化的 Qwen 模型，用于训练
    model=model_sft,
    # 传入 Bespoke - Stratos - 17k 数据集的训练集部分，作为训练数据
    train_dataset=dataset_sft,
    # 传入之前初始化好的分词器，用于处理文本数据
    tokenizer=tokenizer,
    # 传入之前定义好的训练参数，配置训练过程
    args=training_args,
    # 指定数据集中包含文本的字段名，这对于监督微调很重要
    dataset_text_field="conversations",
    # 启用数据打包，以提高训练效率
    packing=True,
    # 设置最大序列长度，用于控制输入序列的长度
    max_seq_length=4096
)

# 启动监督微调（SFT）训练循环
# 调用 sft_trainer 的 train 方法开始训练，训练结果存储在 sft_train_result 中
# 训练结果通常包含训练过程中的统计信息，如损失值、训练步数等
sft_train_result = sft_trainer.train()
```

运行此代码后，您将看到 SFT 训练过程开始。它看起来与 GRPO 训练输出类似，显示每个记录步骤的损失和学习率。
```
...
INFO:__main__:Step 10: Loss = ..., Learning Rate = ...
INFO:__main__:Step 20: Loss = ..., Learning Rate = ...
...
```

与 GRPO 一样，训练时间取决于硬件和所选的epoch。在本例中，我们使用的仍然是一个小型模型，而且只有 1 个epoch，因此训练时间应该是相当快的。