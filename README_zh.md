# Triple Uni 树洞话题分类器

[English](README.md)

## 简介

[**Triple Uni**](https://tripleuni.com/) 是香港大学、香港中文大学和香港科技大学三校的匿名树洞交流平台。

**Triple Uni 树洞话题分类器** 是一个基于 Python 和 PyTorch 的机器学习项目，旨在自动识别和分类平台内的树洞。该项目使用 Softmax 分类器，通过树洞内容的嵌入表示来预测树洞的话题。

你可以在下方示例中在线体验我们的模型，或亲自部署并尝试。

## 数据集

本项目使用 Triple Uni 联校编号 `uni_post_id` 为 1 ～ 500000 的树洞作为数据集，使用 OpenAI 的 [text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings) 模型对树洞内容进行嵌入。

### 数据集涵盖的树洞范围

|学校名称|本校树洞编号|树洞发布时间|
| ------ | ------ | ------ |
|香港大学|1 ～ 253507|2020-10-24 ～ 2022-12-19|
|香港中文大学|1 ～ 190662|2020-08-23 ～ 2022-12-19|
|香港科技大学|1 ～ 55857|2020-01-08 ～ 2022-12-19|

### 数据集格式

每个数据文件 `datasets/post_data_{i}.parquet` 包含联校编号 `uni_post_id` 为 `10000 * i + 1` ～ `10000 * i + 10000` 的10000条树洞。

|字段名|类型|说明|
| ------ | ------ | ------ |
|uni_post_id|int64|树洞的联校编号|
|post_topic|string|树洞的话题|
|embedding|float\[1536\]|树洞正文的1536维嵌入|

### 数据清理

由于各校树洞话题不完全相同，我们将目标话题类别选取为**随写**、**情感**、**学业**、**求职**和**交易**。

对数据进行清理，将**跳蚤**修改为相近类别的**交易**，并剔除所有其他话题。

## 模型架构

请阅读[报告](report.pdf)以了解更多。

## 效果展示

### 分类示例

|树洞内容|模型输出|
| ------ | ------ |
|感觉今天学校好少人 是都还没回来吗|随写|
|不是我不想脱单，只是真的没遇到什么很喜欢的异性罢了。|情感|
|各位申请美国的postgrad的时候有考托福吗？因为我看很多学校说本科是用英语就不需要交托福|学业|
|大摩实习值得去吗？主要做的机构业务|求职|
|出手持蒸汽熨斗 小小的不占地方 没用几次 学校地铁口交收|交易|

### 在线示例

点击[链接](http://tree-hole-judge.tripleuni.com/)，在线体验我们的模型。

## 部署

克隆仓库到本地

```
git clone https://github.com/lststar/Triple-Uni-Topic-Classifier.git
```

安装所需环境

```
pip install -r requirements.txt
```

[下载数据集](https://drive.google.com/drive/u/1/folders/1VVpugGyS-9-mhkvHh4wTX-feFAt3hEVP)

使用 `train.ipynb` 训练模型

在 `main.py` 中填写 `openai_api_key`，并使用以下命令运行 Web UI
```
streamlit run main.py
```

## 支持与联系

如果你对此项目有任何疑问或补充，请提出 Issue 或发送邮件至 **v@tripleuni.com** 。