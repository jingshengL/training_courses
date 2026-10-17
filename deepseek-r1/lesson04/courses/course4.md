推理步骤 奖励有点聪明。我们希望鼓励模型展示其**“思考过程 ”**。因此，我们要奖励它包含*看起来*像推理步骤的东西。


<img src='https://cdn-images-1.medium.com/max/5406/1*hx0sAVnY58WOYw6rGF64ug.png' width=800px>

我们将寻找通常出现在分步推理中的关键词和模式，例如：


* 步骤 1、步骤 2 等
 
* 编号列表，如 1、2

* 要点，如 - 或 *

* 过渡词，如 “首先”、“其次”、“接下来”、"最后
 
其中包含的越多，奖励就越好。这就好比为展示自己的工作给分！

让我们来编写这个推理鼓励功能的代码：
```python
def reasoning_steps_reward(completions, **kwargs):
    r"""
    奖励函数，用于鼓励清晰的逐步推理。
    它会查找诸如 "Step 1:"、编号列表、项目符号以及过渡词等模式。
    """
    # 定义正则表达式模式，用于查找推理步骤的指示符
    # 该模式可以匹配以下几种情况：
    # 1. "Step X:" 格式，其中 X 是数字
    # 2. 以数字加句号开头的行（编号列表）
    # 3. 以换行符加连字符开头的行（项目符号列表）
    # 4. 以换行符加星号开头的行（项目符号列表）
    # 5. 过渡词，如 "First,"、"Second,"、"Next,"、"Finally,"
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"

    # 从每个完成项中提取内容
    # completions 是一个列表，每个元素通常是一个列表，其中第一个元素是包含 "content" 键的字典
    # 这里通过列表推导式提取出每个完成项的内容
    completion_contents = [completion[0]["content"] for completion in completions]

    # 统计每个完成项中推理步骤指示符的数量
    # 使用 re.findall 函数结合 re.MULTILINE 标志，在多行模式下查找所有匹配的模式
    # 并计算匹配结果的数量
    matches = [len(re.findall(pattern, content, re.MULTILINE))
               for content in completion_contents]

    # 奖励值与推理步骤的数量成正比，最大奖励值为 1.0
    # 这里使用了一个“魔法数字” 3，意味着至少有 3 个推理步骤才能获得全额奖励
    # 通过 min 函数确保奖励值不会超过 1.0
    return [min(1.0, count / 3) for count in matches]
```

我们创建了一个更复杂的 regex 模式。它会查找我们上面列出的所有推理指标。

我们使用 re.findall 来查找每个内容中与我们的模式**匹配的内容。然后，“len(re.findall(...)) ”会给出这些指标的**数量。

奖励的计算公式为 min(1.0,count/3)。这意味着

* 如果找到 3 个或更多推理指标（count >= 3），奖励为 1.0（最大奖励）。

* 如果找到的推理指标较少（例如，计数 = 1 或 2），则获得*部分*奖励（如 1/3 或 2/3）。

* 如果找不到（计数 = 0），奖励为 0.0。

这里的 / 3 是一个神奇的数字。我们的意思是"**以 3 步左右的推理步骤为目标，获得全部奖励**" 如果你想鼓励更多或更少的步骤，可以调整这个数字。