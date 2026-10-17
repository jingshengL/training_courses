重复惩罚奖励的目的是阻止我们的模型陷入循环和重复。我们希望它能产生新颖、多样的推理和答案，而不是一遍又一遍地复制粘贴相同的语句！


<img src='https://cdn-images-1.medium.com/max/8608/1*9jBhiz-rI_fRGa77g9RZtQ.png' width=800px>

如果模型使用相同的词语序列（n-grams）的次数过多，该奖励函数就会对模型进行惩罚。在我们的例子中，我们将使用大小为 3 的 n-gram（三格律），但您也可以调整。

如果模型经常重复，就会得到负奖励（惩罚）。如果模型更加多样化并避免重复，惩罚就会减少。

让我们执行代码来惩罚重复：
```python
def get_repetition_penalty_reward(ngram_size: int = 3, max_penalty: float = -0.1):
    """
    返回一个重复惩罚奖励函数。对生成文本中 n - 元组的重复情况进行惩罚。
    """
    # 检查最大惩罚值是否为正数，如果是则抛出错误
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} 不应该为正数")

    def zipngram(text: str, ngram_size: int):
        """辅助函数，用于从文本中生成 n - 元组。"""
        # 将文本转换为小写并按单词分割
        words = text.lower().split()
        # 通过 zip 函数生成 n - 元组
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs):
        """
        重复惩罚奖励函数。
        """
        # 从每个完成项中提取内容
        contents = [completion[0]["content"] for completion in completions]
        # 用于存储每个完成项的奖励值
        rewards = []
        for completion in contents:
            # 如果完成项为空，不进行惩罚，奖励值为 0
            if completion == "":
                rewards.append(0.0)
                continue
            # 如果完成项的单词数量少于 n - 元组的大小，不进行惩罚，奖励值为 0
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            # 使用集合来存储唯一的 n - 元组
            ngrams = set()
            # 用于统计总的 n - 元组数量
            total = 0
            # 生成完成项的 n - 元组
            for ng in zipngram(completion, ngram_size):
                # 将 n - 元组添加到集合中，集合会自动忽略重复项
                ngrams.add(ng)
                # 总 n - 元组数量加 1
                total += 1

            # 计算缩放因子，重复越多，缩放因子越大
            scaling = 1 - len(ngrams) / total
            # 根据缩放因子应用惩罚，得到奖励值
            reward = scaling * max_penalty
            # 将奖励值添加到奖励列表中
            rewards.append(reward)
        return rewards

    # 返回重复惩罚奖励函数
    return repetition_penalty_reward
```

我们的`get_repetition_penalty_reward(...)`创建了一个奖励函数来惩罚重复，其参数包括 ngram_size（默认值为 3，用于三元组）和 max_penalty（负值，例如-0.1）。

辅助函数 “zipngram(text, ngram_size) ”通过将文本转换为小写、拆分成词来生成 n-gram，并使用 "zip(*[words[i:] for i in range(ngram_size)])` 进行高效提取。

在内部，`repetition_penalty_reward(...)` 会计算每次完成的惩罚。如果它是空的或太短，就会得到 0.0 的奖励。

惩罚的缩放比例是 `= 1 - len(ngrams) / total`，其中 total 是 n 个词条的数量，len(ngrams) 是唯一计数。重复次数越多，缩放比例越接近 1，惩罚也就越大。

最终奖励为 scaling * max_penalty，这意味着重复次数越少，惩罚越小，而重复次数越多，负奖励越大。

>我们已经实现了所有五个奖励函数，下面进入下一阶段，定义训练参数