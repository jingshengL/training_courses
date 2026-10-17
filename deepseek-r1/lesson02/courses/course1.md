因此，在介绍技术实现之前，先简单介绍一下 DeepSeek-R1，它并不是从零开始训练的，比如从无到有。相反，他们从已有的一个相当聪明的 LLM [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) 开始，且他们想让它成为一个推理巨星。

![DeepSeek R1 Implementation Quick Overview (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/5872/1*XhE5MVuPfOrrbBxgYtHMsg.png)

为此，他们使用了**强化学习**，简称 RL，即当 LLM 做了有利于推理的事情时，就奖励它，反之则惩罚它。

但这不仅仅是一次简单的培训。这就像一大堆步骤，他们称之为 “管道”。他们首先尝试了纯粹的**RL**，看看推理是否会自己出现，这就是DeepSeek-R1-Zero**，算是一个实验。然后，为了制作**真正的DeepSeek-R1**，他们用不同的阶段让它变得更有组织。他们先给它一些起始数据，让它开始工作，然后进行逻辑推理，再给它更多数据，再进行更多逻辑推理......这就像一步步提升等级一样！

这样做的目的是让这些语言模型更善于思考问题。他们想让它们能够推理，而不是仅仅回答一些问题。


### 模型下载

首先，我们需要先把模型下载一下，下载模型之前我们需要先安装一下所需的库哦，请移步到终端去运行。
```bash
pip install modelscope
```

接下来可以下载所需的模型了！！！

在这里我们已经下载好了可以省略这一步，当然，您想下载其他的模型，也可以自己下载哦！

```python
# coding = utf-8
from modelscope import snapshot_download

snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')
```