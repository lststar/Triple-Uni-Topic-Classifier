# Triple Uni Post Topic Classifier

[中文](README_zh.md)

## Introduction

[Triple Uni](https://tripleuni.com/) is an anonymous communication platform for HKU, CUHK and UST.

The **Triple Uni Post Topic Classifier** is a machine learning project based on Python and PyTorch, designed to automatically identify and classify the topics of posts on the platform. The project uses several models to predict the topics of posts based on their embedded representations.

You can experience our model online in the example below, or deploy and try it yourself.

## Dataset

This project uses posts from the three universities, with `uni_post_id` ranging from 1 to 500,000 as the dataset, using OpenAI's [text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings) model to embed the content of the posts.

### Dataset Coverage

|University Name|ID Range|Posting Time|
| ------ | ------ | ------ |
|The University of Hong Kong (HKU)|1 ～ 253507|2020-10-24 ～ 2022-12-19|
|The Chinese University of Hong Kong (CUHK)|1 ～ 190662|2020-08-23 ～ 2022-12-19|
|The Hong Kong University of Science and Technology (UST)|1 ～ 55857|2020-01-08 ～ 2022-12-19|

### Dataset Format

Each data file `datasets/post_data_{i}.parquet` contains 10,000 posts with `uni_post_id` from `10000 * i + 1` to `10000 * i + 10000`.

|Field Name|Type|Description|
| ------ | ------ | ------ |
|uni_post_id|int64|The inter-university post ID|
|post_topic|string|The topic of the post|
|embedding|float\[1536\]|1536-dimensional embedding of the post content|

### Data Cleaning

Since the topics of posts vary among universities, we chose **Random Thought**, **Emotions**, **Academics**, **Job Hunting**, and **Trading** as target topic categories.

We cleaned the data by renaming **Flea Market** to **Trading** and removing all other topics.

## Neural Network Architecture

Please read [our report](report.pdf) for more detail.

## Performance Demonstration

### Classification Example

|Post Content|Model Output|
| ------ | ------ |
|It feels like there are very few people at school today. Has everyone not returned yet?|Random Thought|
|It's not that I don't want to start dating, it's just that I really haven't met anyone I like a lot.|Emotions|
|Did any of you take the TOEFL when applying for postgraduate programs in the US? I ask because many schools say that if your undergraduate education was in English, you don't need to submit TOEFL scores.|Academics|
|Is an internship at Morgan Stanley worth it? Mainly involved in institutional business.|Job Hunting|
|Selling a handheld steam iron, very compact and barely used. Pick up at the subway station near the school.|Trading|

### Online Example

Click [here](http://tree-hole-judge.tripleuni.com/) to experience our model online.

## Deployment

Clone the repository to your local machine.

```
git clone https://github.com/lststar/Triple-Uni-Topic-Classifier.git
```

Installation required environment.

```
pip install -r requirements.txt
```

[Download Dataset.](https://drive.google.com/drive/u/1/folders/1VVpugGyS-9-mhkvHh4wTX-feFAt3hEVP)

Use `train.ipynb` to train the model

Fill in `openai_api_key` in `main.py`, and run the Web UI with the following command.

```
streamlit run main.py
```

## Support and Contact

If you have any questions or suggestions about this project, please raise an Issue or send an email to **v@tripleuni.com**.