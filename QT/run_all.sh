#!/usr/bin/env bash
# 按课程进度 [1] -> [9] 跑完整个教程（CPU 约 5 分钟）。
# 想单步学习就按 README 第 1 节逐个执行。
set -e
cd "$(dirname "$0")"

echo "========== [1] 量化数学 =========="
python quant_basics.py

echo "========== [2] FP32 baseline =========="
python train_fp32.py

echo "========== [3] 动态 PTQ =========="
python ptq_dynamic.py

echo "========== [4] 静态 PTQ =========="
python ptq_static.py

echo "========== [5] QAT =========="
python qat.py

echo "========== [6] 三条路线总览 =========="
python compare.py

echo "========== [7] LLM 单层方法地图 =========="
python llm_quant.py

echo "========== [8] 迷你 GPT（被试模型） =========="
python tiny_llm.py

echo "========== [9] 整网量化实战 =========="
python llm_case_study.py

echo
echo "全部完成，图表在 outputs/ 下。"
