import streamlit as st
from youtube_comment_downloader import *
import matplotlib.pyplot as plt
from Classes import TwitterDataset, Vocab, RNN

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from nltk import (sent_tokenize as splitter, wordpunct_tokenize as tokenizer)
from collections import Counter

dataset = TwitterDataset('twitter_prep_data_brackets.pickle')
model = joblib.load('abina_model.sav')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_size = 300
hidden_size    = 128
output_size    = 1

def download(message: str):
    comments_text = []
    downloader = YoutubeCommentDownloader()
    try:
        comments = downloader.get_comments_from_url(message, sort_by=SORT_BY_POPULAR)
        for comment in comments:
            comments_text.append(comment['text'])
        comments_df = pd.DataFrame({'text': comments_text})
        result = 'Комментарии получены.'
    except Exception:
        result = 'Ошибка в ссылке'
        comments_df = ''
    return result, comments_df

def predict_sentiment(sentence: str):
    model.eval()
    tokenized = [tokenizer(sentence) for sentence in splitter(sentence)]
    indexed   = [dataset.vocab.sent2idx(tokenized[0])]
    length    = [len(indexed[0])]
    tensor    = torch.LongTensor(indexed).to(device)
    pred      = torch.sigmoid(model(tensor, length))
    return pred.item()

form = st.form(key='input_link')
message = form.text_input(label='Введите ссылку на видео на YouTube')
submit_button = form.form_submit_button(label='Готово')

if submit_button:
    st.write('Подождите, пожалуйста. Обработка может занять несколько минут.')
    comments = download(message)
    st.write(comments[0])
    if comments[0] == 'Комментарии получены.':
        df_comments = comments[1]
        df_comments['sentiment'] = df_comments['text'].apply(predict_sentiment).apply(round, 0)
    st.write('Комментарии обработаны. Спасибо за ожидание!')
    df_comments.groupby('sentiment').count().plot(kind="bar")
    plt.savefig('abr.png')
    st.image('abr.png', caption='Диаграмма распределения позитивных (1) и негативных (0) отзывов')
