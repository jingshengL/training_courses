还记得我们在 R1 Zero 中遇到的混乱推理和语言混合问题吗？SFT 正是为了解决这些问题而设计的。通过在高质量的精炼数据上进行训练，我们正在教授模型：

**清晰的推理风格**： 以易于阅读和理解的方式构建其 “思维”。

**语言一致** ： 在答复中坚持使用一种语言，避免混淆。

我们在 SFT 中使用 Bespoke-Stratos-17k 数据集。正如我们之前看到的，该数据集中了 17,000 个数学和代码问题，其格式非常符合我们的需求。

让我们快速回忆一下 Bespoke-Stratos-17k 中的一个样本：
```python
# 从 bespokelabs 加载 "Bespoke-Stratos-17k" 数据集
# 使用 load_dataset 函数，指定数据集名称 "bespokelabs/Bespoke-Stratos-17k" 和配置名称 "default"
bespoke_rl = load_dataset("bespokelabs/Bespoke-Stratos-17k", "default")

# 访问训练集中的第一个样本
# bespoke_rl['train'] 表示访问训练集，[0] 表示取训练集中的第一个样本
first_sample = bespoke_rl['train'][0]
print(first_sample)

# 以下是输出示例的解释
# 输出是一个字典，包含系统提示信息和对话内容
# 'system' 键对应的值是系统提示，描述了助手的角色等信息
# 'conversations' 键对应的值是一个列表，列表中的每个元素是一个字典，
# 字典包含 'from' 和 'value' 键，分别表示消息的发送方和消息的具体内容
# 示例输出如下：
# {
#   'system': 'Your role as an assistant involves ... ',
#   'conversations': [{'from': 'user', 'value': 'Return your ...'}]
# }
```

该数据集包含系统提示和用户辅助对话，非常适合向我们的模型展示推理对话的样子。

我们将再次使用 trl 库，它能让 SFT 训练变得超级简单。

首先，我们需要设置我们的配置，类似于为 GRPO 所做的配置，但这次是为 SFT 所做的配置。
```python
# 这里假设 ModelConfig 是自定义的配置类
# 从某个模块导入 ModelConfig 类，实际使用时请根据情况修改
# 例如：from your_module import ModelConfig

# 模型和输出配置（与之前相同，或根据需要调整）
# 定义要使用的模型名称
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
# 定义监督微调（SFT）模型的输出目录
OUTPUT_DIR = "data/Qwen-SFT-training"
# 创建输出目录，如果目录已存在则不会报错
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 训练参数 - 与 GRPO 类似，但针对 SFT 进行调整
training_args = TrainingArguments(
    # 训练输出的目录
    output_dir=OUTPUT_DIR,
    # 是否覆盖输出目录中的现有内容
    overwrite_output_dir=True,
    # 训练的轮数，可根据需要调整
    num_train_epochs=1,
    # 每个设备上的训练批次大小
    per_device_train_batch_size=8,
    # 每个设备上的评估批次大小
    per_device_eval_batch_size=16,
    # 梯度累积步数，用于模拟更大的批次大小
    gradient_accumulation_steps=2,
    # 学习率，针对 SFT 进行调整
    learning_rate=2e-5,
    # 热身比例，即训练开始时逐渐增加学习率的比例
    warmup_ratio=0.1,
    # 权重衰减，用于防止过拟合
    weight_decay=0.01,
    # 日志记录的步数间隔
    logging_steps=10,
    # 评估策略，这里设置为不进行评估
    evaluation_strategy="no",
    # 评估的步数间隔
    eval_steps=50,
    # 保存策略，按步数保存模型
    save_strategy="steps",
    # 保存模型的步数间隔
    save_steps=50,
    # 最多保存的模型数量
    save_total_limit=2,
    # 数据加载器的工作进程数
    dataloader_num_workers=2,
    # 随机种子，保证结果可复现
    seed=42,
    # 是否使用 BF16 数据类型进行训练
    bf16=True,
    # 是否将模型推送到 Hugging Face Hub
    push_to_hub=False,
    # 是否启用梯度检查点，用于节省内存
    gradient_checkpointing=True,
    # 不向任何工具报告训练信息
    report_to="none",
    # 启用数据打包以提高效率
    packing=True,
    # 设置最大序列长度
    max_seq_length=4096
)

# 模型配置 - 与之前相同
model_args = ModelConfig(
    # 模型的名称或路径
    model_name_or_path=MODEL_NAME,
    # 模型的版本
    model_revision="main",
    # 模型使用的张量数据类型
    torch_dtype="bfloat16",
    # 是否信任远程代码
    trust_remote_code=True,
    # 注意力机制的实现方式
    attn_implementation="flash_attention_2"
)
```
这些 TrainingArguments 和 ModelConfig 与我们在 GRPO 中使用的非常相似，但做了一些更适合 SFT 的调整（比如学习率略有不同，重要的是 packing=True 和 max_seq_length=4096，以便在更长的序列上进行高效训练）。