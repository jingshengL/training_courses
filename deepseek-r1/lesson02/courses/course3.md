
既然我们已经选择了基础模型，接下来就需要了解训练 LLM 的基本 RL 设置是如何工作的。

DeepSeek R1的起点是（DeepSeek V3）基础模型，而我们的起点是Qwen2.5-0.5B-Instruct。我所说的起点是指**它已创建了DeepSeek R1零版**，即在创建最终版本之前有一些错误的初始版本。

初始版本（R1 零）是通过强化学习（DeepSeek v3/Qwen2.5-0.5B）创建的，在强化学习中，（DeepSeek v3/Qwen2.5-0.5B）充当了一个 RL 代理（采取行动的演员）。让我们先来看看它是如何工作的。

![Qwen 2.5 as an agent workflow (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/5410/1*S6YIXu1vIVmQFl-DgRFktg.png)

RL 代理（DeepSeek V3/Qwen2-0.5B）首先会采取**行动**，这意味着它会针对给定问题生成一个答案和一些推理，并将其放入**环境**中。在这种情况下，环境就是推理任务本身。

在采取行动后，环境会反馈一个**奖励**。这个奖励就像是反馈，它会告诉我们的基础模型（DeepSeek V3/Qwen2-0.5B）它的行动有多好。一个积极的奖励意味着它做对了什么，也许是答对了，也许是推理得很好。这个反馈信号会传回我们的基础模型，帮助它学习并调整未来的行动方式，以获得更好的奖励。
> 在下一章节中，我们将更详细地讨论这种方法