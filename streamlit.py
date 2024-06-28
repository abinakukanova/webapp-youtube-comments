import nltk
nltk.download('all')
import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from transformers import pipeline

pipe = pipeline("summarization", model="IlyaGusev/mbart_ru_sum_gazeta")

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



def main():
    st.title("Суммаризатор комментариев")
    comments_df = pd.read_csv('shulman_comments.csv')
    all_comments = comments_df['text'].to_list()
    grouped_comments = group_comments(all_comments, max_tokens=400)
    comments_per_page = 3
    page_number = st.session_state.get("page_number", 1)

    if st.button("Получить краткое содержание"):
        st.session_state.page_number = 1  # сбрасываем номер страницы при каждом новом нажатии кнопки
        with st.spinner("Подождите пожалуйста! Комментарии обрабатываются..."):
            display_comments(grouped_comments, st.session_state.page_number, comments_per_page)

    if "page_number" in st.session_state and st.session_state.page_number * comments_per_page < len(grouped_comments):
        if st.button("Ещё"):
            st.session_state.page_number += 1
            display_comments(grouped_comments, st.session_state.page_number, comments_per_page)  


@st.cache_data 
def display_comments(grouped_comments, page_number, comments_per_page):
    start_index = (page_number - 1) * comments_per_page
    end_index = start_index + comments_per_page

    for i in range(start_index, min(end_index, len(grouped_comments))):
        summary =  pipe(grouped_comments[i])[0]['summary_text']
        st.write(f"*Группа комментариев {i+1}:*")
        st.write(summary)

if __name__ == "__main__":
    main() 
