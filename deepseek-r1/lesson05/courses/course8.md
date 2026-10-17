完成 SFT 后，我们保存新的微调模型 (R1)。
```python
# 保存训练好的监督微调（SFT）模型

# 定义训练好的 SFT 模型的保存路径，这里与之前定义的输出目录 OUTPUT_DIR 相同
TRAINED_SFT_MODEL_PATH = "data/Qwen-SFT-training"

# 保存分词器
# 使用 save_pretrained 方法将分词器保存到指定的路径
# 这样后续可以从该路径加载分词器用于文本处理
tokenizer.save_pretrained(TRAINED_SFT_MODEL_PATH)

# 保存训练好的模型
# 调用 sft_trainer 的 save_model 方法，将训练好的模型保存到指定路径
# 该方法会保存模型的权重和相关配置信息
sft_trainer.save_model(TRAINED_SFT_MODEL_PATH)

# 打印提示信息，告知训练好的 SFT 模型已保存到指定路径
print(f"SFT Trained model saved to {TRAINED_SFT_MODEL_PATH}")
```
SFT 阶段1部分到此为止！现在，我们已经采用了我们的基础模型，向它展示了许多良好推理的例子，并对其进行了微调，使其更能做出清晰、有条理的回答。
> 这个使用 SFT 的微调模型就是我们所说的 SFT 阶段 1 之后的 R1模型阶段2

SFT 阶段2 的步骤，尤其是 RL 阶段和拒绝采样，在 Python 中从头开始实现非常复杂。注重理论理解是理解整个过程的关键。