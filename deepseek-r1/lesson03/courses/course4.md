
现在，我们要编写一个配置，在这里我们可以微调*奖励函数*的实际工作方式。因此，让我们来定义这个配置类：

```python
# 定义用于奖励函数参数的GRPOScriptArguments类
from dataclasses import dataclass, field

@dataclass
class GRPOScriptArguments:
    """
    用于GRPO训练的脚本参数，具体与奖励函数相关。
    """

    # 奖励函数列表，默认包含 "accuracy" 和 "format"
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            # 帮助信息，说明可能的奖励函数值
            "help": "奖励函数列表。可能的值: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty'"
        },
    )
    # 错误答案余弦缩放的最小奖励值
    cosine_min_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "错误答案余弦缩放的最小奖励值"}
    )
    # 错误答案余弦缩放的最大奖励值
    cosine_max_value_wrong: float = field(
        default=-0.1,
        metadata={"help": "错误答案余弦缩放的最大奖励值"}
    )
    # 正确答案余弦缩放的最小奖励值
    cosine_min_value_correct: float = field(
        default=0.8,
        metadata={"help": "正确答案余弦缩放的最小奖励值"}
    )
    # 正确答案余弦缩放的最大奖励值
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "正确答案余弦缩放的最大奖励值"}
    )
    # 余弦缩放的最大长度
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "余弦缩放的最大长度"}
    )
    # 重复惩罚奖励的n-gram数量
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "重复惩罚奖励的n-gram数量"}
    )
    # 重复惩罚奖励的最大（负）惩罚值
    repetition_max_penalty: float = field(
        default=-0.1,
        metadata={"help": "重复惩罚奖励的最大（负）惩罚值"}
    )
```

我们的 `@dataclass` 装饰器可轻松创建用于存储数据的类。而 GRPOScriptArguments 类则用于保存奖励设置。

reward_funcs 列表决定了要使用哪些奖励，从[“准确度”、“格式”]开始，但你还可以添加更多，如 “reasoning_steps”、“cosine”、“repetition_penalty”。

有些设置可以控制余弦放大奖励（cosine_scaled_reward）和重复惩罚奖励（repetition_penalty_reward）的工作方式，让你可以调整奖励的发放方式。

接下来是转换器库中的 TrainingArguments。这是***主要的***配置对象，控制着**几乎整个**训练过程的。
```python
# 从transformers库中定义TrainingArguments类的实例
from transformers import TrainingArguments

# 假设OUTPUT_DIR已经定义，例如：
OUTPUT_DIR = "results"

# 定义训练参数
training_args = TrainingArguments(
    # 检查点和日志的输出目录
    output_dir=OUTPUT_DIR,
    # 是否覆盖输出目录中的已有内容
    overwrite_output_dir=True,
    # 总的训练轮数
    num_train_epochs=1,
    # 训练期间每个设备的批次大小
    per_device_train_batch_size=8,
    # 评估时每个设备的批次大小
    per_device_eval_batch_size=16,
    # 梯度累积步数，用于模拟更大的批次大小
    gradient_accumulation_steps=2,
    # AdamW优化器的初始学习率
    learning_rate=5e-5,
    # 训练步骤中线性热身的比例
    warmup_ratio=0.1,
    # 除偏置和LayerNorm权重外，对所有层应用权重衰减
    weight_decay=0.01,
    # 每X个更新步骤记录一次日志
    logging_steps=10,
    # 评估策略，按步骤进行评估
    evaluation_strategy="steps",
    # 评估和记录日志的步数间隔
    eval_steps=50,
    # 保存检查点的策略，按步骤保存
    save_strategy="steps",
    # 每X个更新步骤保存一次检查点
    save_steps=50,
    # 限制检查点的总数，删除较旧的检查点
    save_total_limit=2,
    # 用于数据加载的子进程数量
    dataloader_num_workers=2,
    # 随机种子，用于保证结果的可重复性
    seed=42,
    # 是否使用混合精度BFP16训练
    bf16=True,
    # 是否将最终模型推送到Hugging Face Hub
    push_to_hub=False,
    # 是否启用梯度检查点以节省内存
    gradient_checkpointing=True,
    # 不向任何地方报告训练结果
    report_to="none"
)
```

最后，我们需要一个模型调试-ModelConfig-。在这里，我们要对模型本身进行特定设置，比如使用哪种预训练模型、使用哪种数据类型（如 bfloat16）、是否信任远程代码等等。

让我们定义我们的 ModelConfig：
```python
@dataclass
class ModelConfig:
    """
    模型的配置类。
    """
    # 预训练模型的路径或者来自huggingface.co/models的模型标识符
    model_name_or_path: str = field(
        default=MODEL_NAME, 
        metadata={"help": "预训练模型的路径或者来自huggingface.co/models的模型标识符"}
    )
    # 要使用的具体模型版本（可以是分支名、标签名或提交ID）
    model_revision: Optional[str] = field(
        default="main", 
        metadata={"help": "要使用的具体模型版本（可以是分支名、标签名或提交ID）"}
    )
    # 覆盖默认的 `torch_dtype`，并在此数据类型下加载模型
    torch_dtype: Optional[str] = field(
        default="bfloat16", 
        metadata={"help": "覆盖默认的 `torch_dtype`，并在此数据类型下加载模型"}
    )
    # 加载模型和分词器时是否信任远程代码
    trust_remote_code: bool = field(
        default=True, 
        metadata={"help": "加载模型和分词器时是否信任远程代码"}
    )
    # 要使用的注意力机制实现方式，可选 'flash_attention_2' 或 None
    attn_implementation: Optional[str] = field(
        default="flash_attention_2", 
        metadata={"help": "要使用的注意力机制实现方式。可选 'flash_attention_2' 或 None"}
    )
```

我们的**ModelConfig**类包含关键设置，包括模型名称或路径，默认为**Qwen/Qwen2.5-7B-Instruct*。为了提高效率，我们使用 torch_dtype=“bfloat16”，并设置 trust_remote_code=True 以确保远程加载安全。此外，还启用了 attn_implementation=“flash_attention_2”，以便在支持的情况下加快训练速度。

现在，我们需要为这些配置类**创建**实例，以便使用它们：
```python
# 实例化配置对象
# 实例化GRPOScriptArguments类，得到包含奖励函数相关参数配置的对象
script_args = GRPOScriptArguments()
# 实例化ModelConfig类，得到包含模型相关参数配置的对象
model_args = ModelConfig()
```

接下来，我们需要获取奖励函数列表以及我们希望在训练中使用的任何 “回调”。

回调就像一个小帮手，可以在训练过程的不同阶段做一些事情（如记录进度、保存模型等）。现在，我们只使用一个简单的日志回调。

在一个地方获取我们的奖励函数。
```python
# 基于脚本参数获取奖励函数的工具函数
def get_reward_functions(script_args):
    """
    根据脚本参数返回一个奖励函数列表。
    """
    # 用于存储奖励函数的列表
    reward_funcs_list = []
    # 奖励函数注册表，将奖励函数名称映射到具体的奖励函数
    reward_funcs_registry = {
        # 假设 accuracy_reward 函数在之前的步骤中已经定义
        "accuracy": accuracy_reward,
        # 假设 format_reward 函数在之前的步骤中已经定义
        "format": format_reward,
        # 假设 reasoning_steps_reward 函数已经定义
        "reasoning_steps": reasoning_steps_reward,
        # 假设 get_cosine_scaled_reward 函数已经定义
        "cosine": get_cosine_scaled_reward(
            # 从脚本参数中获取错误答案余弦缩放的最小奖励值
            min_value_wrong=script_args.cosine_min_value_wrong,
            # 从脚本参数中获取错误答案余弦缩放的最大奖励值
            max_value_wrong=script_args.cosine_max_value_wrong,
            # 从脚本参数中获取正确答案余弦缩放的最小奖励值
            min_value_correct=script_args.cosine_min_value_correct,
            # 从脚本参数中获取正确答案余弦缩放的最大奖励值
            max_value_correct=script_args.cosine_max_value_correct,
            # 从脚本参数中获取余弦缩放的最大长度
            max_len=script_args.cosine_max_len,
        ),
        # 假设 get_repetition_penalty_reward 函数已经定义
        "repetition_penalty": get_repetition_penalty_reward(
            # 从脚本参数中获取重复惩罚奖励的 n-gram 数量
            ngram_size=script_args.repetition_n_grams,
            # 从脚本参数中获取重复惩罚奖励的最大（负）惩罚值
            max_penalty=script_args.repetition_max_penalty,
        ),
    }

    # 遍历脚本参数中指定的奖励函数名称
    for func_name in script_args.reward_funcs:
        # 检查指定的奖励函数名称是否在注册表中
        if func_name not in reward_funcs_registry:
            # 如果不在注册表中，抛出值错误
            raise ValueError(f"奖励函数 '{func_name}' 未在注册表中找到。")
        # 将对应的奖励函数添加到奖励函数列表中
        reward_funcs_list.append(reward_funcs_registry[func_name])

    # 返回奖励函数列表
    return reward_funcs_list
```
我们的回调函数，用于跟踪损失和其他重要信息。
```python
# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

class LoggingCallback(TrainerCallback):
    """
    一个简单的回调类，用于在特定步骤记录训练信息。
    """
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # 检查当前全局步数是否是日志记录步数的整数倍
        if state.global_step % args.logging_steps == 0:
            # 从日志历史记录中获取最新的损失值，如果没有则为 None
            loss = state.log_history[-1].get('loss', None)
            # 从日志历史记录中获取最新的学习率，如果没有则为 None
            learning_rate = state.log_history[-1].get('learning_rate', None)
            # 使用日志记录器记录当前步数、损失值和学习率
            logger.info(f"Step {state.global_step}: Loss = {loss}, Learning Rate = {learning_rate}")

def get_callbacks(training_args, model_args, script_args):
    """
    返回训练期间要使用的回调函数列表。
    目前，仅包含 LoggingCallback。你可以扩展此函数以添加更多回调。
    """
    # 实例化 LoggingCallback 类，将其添加到回调列表中
    callbacks = [LoggingCallback()]
    # 返回回调列表
    return callbacks
```

最后，初始化这些函数。
```python
# 获取奖励函数和回调函数

# 调用 get_reward_functions 函数，传入之前实例化的脚本参数对象 script_args
# 该函数会根据脚本参数中指定的奖励函数名称，从奖励函数注册表中获取对应的奖励函数
# 并将这些奖励函数组成一个列表返回，赋值给 reward_functions 变量
reward_functions = get_reward_functions(script_args)

# 调用 get_callbacks 函数，传入训练参数对象 training_args、模型参数对象 model_args 和脚本参数对象 script_args
# 该函数目前仅返回一个包含 LoggingCallback 实例的列表，这个回调函数用于在特定训练步骤记录训练信息
# 将返回的回调函数列表赋值给 callbacks 变量
callbacks = get_callbacks(training_args, model_args, script_args)
```