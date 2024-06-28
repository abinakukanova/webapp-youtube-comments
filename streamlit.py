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
import nltk
nltk.download('all')
from nltk import (sent_tokenize as splitter, wordpunct_tokenize as tokenizer)
from collections import Counter
from nltk.tokenize import word_tokenize
from transformers import pipeline

pipe = pipeline("summarization", model="IlyaGusev/mbart_ru_sum_gazeta")

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

def group_comments(comments, max_tokens=400):
    """
    Объединяет комментарии в строки, ограничивая их длину 400 токенами.
    """
    grouped_comments = []
    current_tokens = []
    current_comment = ""

    for comment in comments:
        tokens = word_tokenize(comment)
        if len(current_tokens) + len(tokens) <= max_tokens:
            current_tokens.extend(tokens)
            current_comment += comment + " "
        else:
            grouped_comments.append(current_comment.strip())
            current_tokens = tokens
            current_comment = comment + " "

    # Добавляем последний комментарий
    if current_comment:
        grouped_comments.append(current_comment.strip())

    return grouped_comments

def summarize_group(group):
    """Суммаризирует одну группу комментариев"""
    summary = pipe(group, max_length=100, min_length=30)[0]['summary_text']
    return summary

def display_summaries(grouped_comments, page_number, comments_per_page):
    start_index = (page_number - 1) * comments_per_page
    end_index = start_index + comments_per_page

    for i in range(start_index, min(end_index, len(grouped_comments))):
        st.write(f"**Группа комментариев {i+1}:**")
        summary = summarize_group(grouped_comments[i])
        st.write(summary)

def main():
    st.title("Анализ комментариев с YouTube")
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

            # Суммаризация
            all_comments = df_comments['text'].to_list()
            grouped_comments = group_comments(all_comments, max_tokens=400)

            # Отображение сводок
            comments_per_page = 3
            page_number = st.session_state.get("page_number", 1)

            if st.button("Получить краткое содержание"):
                st.session_state.page_number = 1
                with st.spinner("Подождите пожалуйста! Комментарии обрабатываются..."):
                    display_summaries(grouped_comments, st.session_state.page_number, comments_per_page)

            if "page_number" in st.session_state and st.session_state.page_number * comments_per_page < len(grouped_comments):
                if st.button("Ещё"):
                    st.session_state.page_number += 1
                    display_summaries(grouped_comments, st.session_state.page_number, comments_per_page)

if __name__ == "__main__":
    main()
