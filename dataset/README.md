# 训练数据

每个文件采用 UTF-8 编码的 JSONL 格式，每行一个完整 JSON 对象。四行预训练样例可以直接打开。 [examples/pretrain.jsonl](examples/pretrain.jsonl)

| 阶段 | 每条记录的字段 | 加载器 |
| --- | --- | --- |
| 预训练 | `text` 字符串 | `PretrainDataset` |
| SFT / 推理蒸馏 | `conversations` 消息列表 | `SFTDataset` |
| GRPO | `conversations` 消息列表，末轮为助手回复 | `RLAIFDataset` |
| DPO | `chosen` 与 `rejected`，分别是一组完整对话 | `DPODataset` |

**预训练**用同一段文本构造输入和右移一位的目标。

```json
{"text":"月球绕着地球运行。"}
```

**监督微调**使用分词器的对话模板，并通过损失掩码选择参与训练的位置。推理蒸馏沿用相同的外层结构，助手回复可以包含 `<think>` 和 `<answer>` 标签。

```json
{"conversations":[{"role":"user","content":"什么是词表？"},{"role":"assistant","content":"词表记录分词器可识别的 token 及其编号。"}]}
```

**GRPO** 从交替排列的用户与助手消息中提取提示，用策略模型重新采样回答，再由奖励模型打分。

**DPO** 使用同一提示下的一对回答。

```json
{"chosen":[{"role":"user","content":"一小时有多少分钟？"},{"role":"assistant","content":"60 分钟。"}],"rejected":[{"role":"user","content":"一小时有多少分钟？"},{"role":"assistant","content":"10 分钟。"}]}
```

完整数据可按阶段下载。将文件放入本目录，或给训练脚本传入 `--data_path`。数据加载器会把 JSONL 读入内存，初次试验可先使用小文件。 [历史数据版本](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/84983ed4dec7836d240577760c1d6be5d4cabcf9) · [加载器实现](lm_dataset.py)
