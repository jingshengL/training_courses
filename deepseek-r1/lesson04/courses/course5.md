余弦比例奖励则更先进一些。它鼓励正确答案的*简洁性*，而对较长的错误答案则*不苛刻*。


<img src='https://cdn-images-1.medium.com/max/7094/1*WmG8r1OVeU4R3jObAy0yCg.png' width=800px>

可以这样想：

**对于正确答案：** 我们希望奖励*短*、更直接的答案，而不是长篇大论的答案。简短、正确的答案往往更好。

**对于错误答案：**简短、错误的答案可能比至少**尝试**推理的较长、错误的答案更糟糕。因此，我们希望对简短的错误答案的惩罚要比对较长的错误答案的惩罚更重。

让我们来看看实现这种巧妙缩放的代码：
```python
import math

# 实现余弦缩放奖励函数
def get_cosine_scaled_reward(
    # 错误答案的最小奖励值，默认为 -0.5
    min_value_wrong: float = -0.5,
    # 错误答案的最大奖励值，默认为 -0.1
    max_value_wrong: float = -0.1,
    # 正确答案的最小奖励值，默认为 0.8
    min_value_correct: float = 0.8,
    # 正确答案的最大奖励值，默认为 1.0
    max_value_correct: float = 1.0,
    # 最大长度，用于计算进度，默认为 1000
    max_len: int = 1000,
):
    """
    返回一个余弦缩放奖励函数。此函数会根据完成长度对准确率奖励进行缩放。
    较短的正确答案会获得更高的奖励，较长的错误答案会减少惩罚。
    """
    def cosine_scaled_reward(completions, solution, accuracy_rewards, **kwargs):
        """
        余弦缩放奖励函数，根据完成长度调整准确率奖励。
        """
        # 从每个完成项中提取内容
        contents = [completion[0]["content"] for completion in completions]
        # 用于存储每个完成项的最终奖励值
        rewards = []

        # 遍历每个完成项的内容、对应的标准答案和准确率奖励
        for content, sol, acc_reward in zip(contents, solution, accuracy_rewards):
            # 生成答案的长度
            gen_len = len(content)
            # 计算当前生成答案长度相对于最大长度的进度
            progress = gen_len / max_len
            # 根据进度计算余弦值
            cosine = math.cos(progress * math.pi)

            # 判断答案是否正确
            if acc_reward > 0.5:  # 假设准确率奖励对于正确答案约为 1.0
                # 正确答案的最小奖励值
                min_value = min_value_correct
                # 正确答案的最大奖励值
                max_value = max_value_correct
            else:  # 错误答案
                # 注意这里的交换，错误答案的最小惩罚值（实际是最大奖励值）
                min_value = max_value_wrong
                # 错误答案的最大惩罚值（实际是最小奖励值）
                max_value = min_value_wrong

            # 余弦缩放公式，根据余弦值和最大最小奖励值计算最终奖励
            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            # 将计算得到的奖励值添加到奖励列表中
            rewards.append(float(reward))
        # 返回所有完成项的最终奖励值列表
        return rewards
    # 返回内部定义的余弦缩放奖励函数
    return cosine_scaled_reward
```

`get_cosine_scaled_reward(...)` 生成一个用于训练的奖励函数，使用 min_value_wrong/max_value_wrong（错误答案的惩罚范围）和 min_value_correct/max_value_correct（正确答案的奖励范围）等参数自定义缩放。max_len 设置缩放的最大长度。

在 `cosine_scaled_reward(...)`内部，我们根据完成度、解法和准确度奖励计算奖励。

它会计算 gen_len，将其归一化为进度 `= gen_len / max_len`，并得出一个余弦值，该值从 1 开始（短答案），递减到 -1 （长答案）。

如果 acc_reward > 0.5，它就会使用正确的奖励范围，否则就会使用错误的奖励范围，但会交换最小/最大值，以减少对较长的错误答案的惩罚。