import numpy as np
import torch
from torch import nn
from openai import OpenAI
import joblib
import streamlit as st

model_name1 = 'topic-classifier-random-forest'
model_name2 = 'topic-classifier-softmax'
model_name3 = 'topic-classifier-neural-network'
openai_api_key = '' # TODO
openai_model = 'text-embedding-ada-002'

topic_list = ['交易', '学业', '情感', '求职', '随写']
topic_translation = {
    '交易': 'Trading',
    '学业': 'Academics',
    '情感': 'Emotions',
    '求职': 'Job Hunting',
    '随写': 'Random Thoughts'
}

class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SoftmaxClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

class NeuralNetworkClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(NeuralNetworkClassifier, self).__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(128, 32)
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(32, 16)
        self.batchnorm3 = nn.BatchNorm1d(16)
        self.relu3 = nn.ReLU()
        self.output = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        x = self.output(x)
        return x

client = OpenAI(api_key = openai_api_key)
model1 = joblib.load(f'models/{model_name1}.joblib')
model2 = torch.load(f'models/{model_name2}.pth')
model2.eval()
model3 = torch.load(f'models/{model_name3}.pth')
model3.eval()

st.set_page_config(
    page_title = 'Triple Uni - Post Topic Classifier',
    layout = 'centered',
)
st.title('Triple Uni - Post Topic Classifier')
st.write('This is a live demo for https://github.com/lststar/Triple-Uni-Topic-Classifier .')
st.write('This website uses three machine learning models to determine the topic of a post based on the content you enter.')
st.write('The post will be classified to one of the following: Trading, Academics, Emotions, Job Hunting and Random Thoughts.')

def main():
    content = st.text_area('Please enter the content to be classified:', height = 200)
    
    if st.button('Classify!') and content:
        with st.spinner('Classifying...'):
            embedding = client.embeddings.create(input=[content], model=openai_model).data[0].embedding
            
            outputs1 = model1.predict(np.array(embedding).reshape(1, -1))
            predicted1 = outputs1[0]
            predicted_topic1 = topic_translation[topic_list[int(predicted1)]]
            
            outputs2 = model2(torch.tensor(embedding).unsqueeze(0))
            _, predicted2 = torch.max(outputs2.data, 1)
            predicted_topic2 = topic_translation[topic_list[int(predicted2)]]
            
            outputs3 = model3(torch.tensor(embedding).unsqueeze(0))
            _, predicted3 = torch.max(outputs3.data, 1)
            predicted_topic3 = topic_translation[topic_list[int(predicted3)]]

            st.info(f"""
                    Random Forest: {predicted_topic1}\r\n
                    Softmax: {predicted_topic2}\r\n
                    Neural Network: {predicted_topic3}\r\n
                    """)
            print(f"`{content}`\nRandom Forest: {predicted_topic1}\nSoftmax: {predicted_topic2}\nNeural Network: {predicted_topic3}")

if __name__ == '__main__':
    main()