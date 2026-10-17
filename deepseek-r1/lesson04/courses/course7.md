这是实际驱动 GRPO 训练的引擎。我们需要初始化它，为它提供我们准备好的所有组件：模型、奖励函数、训练参数、数据集和回调！

让我们初始化 GRPOTrainer：
```python
# 从 TrainingArguments 创建 GRPOConfig
# 利用 training_args.to_dict() 将 TrainingArguments 对象转换为字典，
# 再使用 ** 解包操作将字典中的键值对作为参数传递给 GRPOConfig 类，
# 以此初始化 GRPOConfig 对象
grpo_config = GRPOConfig(
    **training_args.to_dict(), 
    **{ 
       # 这里移除了 model_init_kwargs
       # 由于我们直接传入了已实例化的 'model' 对象，
       # 所以 GRPOTrainer 不需要 model_init_kwargs 来初始化模型
    }
)

# 初始化 GRPOTrainer
grpo_trainer = GRPOTrainer(
    # 传入已初始化的 Qwen 模型
    model=model,                      
    # 传入之前步骤中获取的奖励函数列表
    reward_funcs=reward_functions,    
    # 传入由 TrainingArguments 创建的 GRPOConfig 对象
    args=grpo_config,                
    # 传入训练数据集
    train_dataset=dataset['train'],   
    # 传入评估数据集
    eval_dataset=dataset['test']         
)
```

现在我们可以开始**训练循环**！只需调用 grpo_trainer 上的 train() 方法即可。
```python
# 启动 GRPO 训练循环
# 调用 grpo_trainer 对象的 train 方法来开始训练过程
# train 方法会根据之前配置好的训练参数、奖励函数、数据集等进行训练
# 训练完成后，会将训练结果存储在 train_result 变量中，
# 这个结果通常包含了训练过程中的一些统计信息，比如损失值、训练步数等
train_result = grpo_trainer.train()
```
运行该单元后，您将看到训练过程开始。
```
...
INFO:__main__:Step 10: Loss = ..., Learning Rate = ...
INFO:__main__:Step 20: Loss = ..., Learning Rate = ...
...
```
训练需要一些时间，但我们设置了 **num_train_epochs = 1**，而且使用的是一个小型模型，因此本例的训练时间应该不会太长。

但对于真实世界的 GRPO DeepSeek R1 Zero 训练，你的训练过程可能需要更多的epochs和步骤。

#### 注：epochs被定义为向前和向后传播中所有批次的单次训练迭代。这意味着1个周期是整个输入数据的单次向前和向后传递。简而言之，epochs指的就是训练过程中所有数据将被“迭代”多少次，这是所有数据迭代训练的轮数。