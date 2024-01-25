import os
# for data handling
import pandas as pd
import numpy as np
# for regular expression
import re
# for text prerprocessing
from nltk.tokenize import word_tokenize
from konlpy.tag import Okt
# for sub-word tokenizing
import sentencepiece as spm
# for making tf-idf, count vector
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# for lda
from sklearn.decomposition import LatentDirichletAllocation
# for matrix factorization
from sklearn.decomposition import NMF
# for ppmi_matrix
from scipy.sparse import csr_matrix
# for similarity
from sklearn.metrics.pairwise import cosine_similarity
# for datetime
from datetime import datetime
# for LLM
from langchain import OpenAI
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.summarize import load_summarize_chain
# for translate
import deepl
# for front
import streamlit as st
# for warining
import warnings
warnings.filterwarnings("ignore")

os.environ["OPENAI_API_KEY"] = "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"

def main():
    data1 = pd.read_csv("./data/merge_data_review.csv") # 한글책 데이터
    data2 = pd.read_csv("./data/translate.csv") # 원서번역 데이터
    data1.dropna(inplace=True) # 결측치 제거
    data2.dropna(inplace=True) # 결측치 제거

    def preprocess(text):
        # 한글,영어,숫자 제외 제거
        text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
        # 공백이 여러개일 경우 하나로 치환
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def ppmi_matrix(count_matrix):
        # 총 단어 수 계산
        total_count = count_matrix.sum()
        # 단어별 총 출현 빈도 계산 
        word_count = count_matrix.sum(axis=0).A1
        # 문서별 총 단어 수 계산
        doc_count = count_matrix.sum(axis=1).A1
        # PPMI 계산
        ppmi_values = np.log(((count_matrix / doc_count[:, np.newaxis]) * total_count) / word_count[np.newaxis, :])
        ppmi_values = np.maximum(0, ppmi_values)  # 음수 값을 0으로 변경
        ppmi_values = np.nan_to_num(ppmi_values)  # NaN 값을 0으로 대체
        return csr_matrix(ppmi_values)
    
    def sp_tokenize(text, sp_processor):
        # SentencePiece를 사용하여 토큰화
        tokens = sp_processor.encode_as_pieces(text)
        return tokens
    
    def clean_text(text):
        # 특수 문자 및 숫자 제거
        text = re.sub(r'[^가-힣a-zA-Z\s]', '', text)
        # 공백으로 단어 분리
        tokens = word_tokenize(text)
        # 불용어 제거
        filtered_tokens = [word for word in tokens if word not in stop_words]
        # 문자열 변환
        text = ' '.join(filtered_tokens)
        return text

    # 모델링에 사용할 열의 경우 추가 전처리
    corpus1 = []
    corpus2 = []
    for _, row in data1.iterrows():
        combined_text = f"{row['title']} {row['index']} {row['book_info']} {row['author_info']}"
        preprocessed_text = preprocess(combined_text)
        words = preprocessed_text.split()
        corpus1.extend(words)

    for _, row in data2.iterrows():
        combined_text = f"{row['translate_title']} {row['translate_index']} {row['translate_book_info']} {row['translate_author']}"
        preprocessed_text = preprocess(combined_text)
        words = preprocessed_text.split()
        corpus2.extend(words)

    # corpus 생성
    with open('./data/corpus.txt', 'w', encoding='utf-8') as f:
        for word in corpus1:
            f.write(word + '\n')
    
    with open('./data/corpus.txt', 'a', encoding='utf-8') as f:
        for word in corpus2:
            f.write(word+ '\n')

    # korean_stop_words = [
    # '의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다'
    # ]

    with open('./data/stopword.txt', 'r', encoding='utf-8') as file:
        korean_stop_words = [line.strip() for line in file]

    stop_words = set(korean_stop_words)

    spm.SentencePieceTrainer.train('--input=./data/corpus.txt --model_prefix=m --vocab_size=2000')

    # subword tokeinizing 모델을 로드
    sp = spm.SentencePieceProcessor(model_file='m.model')
    sp.load('m.model')


    for column in ['book_info', 'title', 'index', 'author_info']:
        data1[column] = data1[column].apply(clean_text)

    for column in ['translate_book_info', 'translate_title', 'translate_index', 'translate_author']:
        data2[column] = data2[column].apply(clean_text)
    
    # CountVecotrize
    count_vectorizer_1 = CountVectorizer()
    count_vectorizer_2 = CountVectorizer()
    count_matrix_1 = count_vectorizer_1.fit_transform(data1['book_info'] + " " + data1['title'] + " " + data1['index'])
    count_matrix_2 = count_vectorizer_2.fit_transform(data2['translate_book_info'] + " " + data2['translate_title'] + " " + data2['translate_index'])

    # TF-IDF Matrix
    tfidf_vectorizer_1 = TfidfVectorizer()
    tfidf_vectorizer_2 = TfidfVectorizer()
    tfidf_matrix_1 = tfidf_vectorizer_1.fit_transform(data1['book_info'] + " " + data1['title'] + " " + data1['index'])
    tfidf_matrix_2 = tfidf_vectorizer_2.fit_transform(data2['translate_book_info'] + " " + data2['translate_title'] + " " + data2['translate_index'])

    # PPMI Matrix
    ppmi_matrix_1 = ppmi_matrix(count_matrix_1)
    ppmi_matrix_2 = ppmi_matrix(count_matrix_2)

    count_matrix_1.toarray()
    count_matrix_2.toarray()
    data1['combined'] = data1['book_info'] + " " + data1['title'] + " " + data1['index']
    data1['tokenized'] = data1['combined'].apply(lambda x: ' '.join(sp_tokenize(x, sp)))

    data2['combined'] = data2['translate_book_info'] + " " + data2['translate_title'] + " " + data2['translate_index']
    data2['tokenized'] = data2['combined'].apply(lambda x: ' '.join(sp_tokenize(x, sp)))

    # LDA 기반 Content base-filtering
    lda_model_1 = LatentDirichletAllocation(n_components=5) # 'n_components' [nlp, cv, rl, rc, other]
    lda_model_2 = LatentDirichletAllocation(n_components=5) 
    lda_matrix_1 = lda_model_1.fit_transform(count_matrix_1)
    lda_matrix_2 = lda_model_2.fit_transform(count_matrix_2)
    
    # 비음수 행렬분해
    nmf_model_1 = NMF(n_components=5)
    nmf_model_2 = NMF(n_components=5)
    nmf_matrix_1 = nmf_model_1.fit_transform(ppmi_matrix_1)
    nmf_matrix_2 = nmf_model_2.fit_transform(ppmi_matrix_2)

    def calculate_time_weight(release_date, current_date=datetime.now()):

        release_date = datetime.strptime(str(release_date), "%Y%m%d")

        days_diff = (current_date - release_date).days

        time_weight = 0.5 ** (days_diff / 365)
        
        return time_weight
    
    def generate_recommendations(data1, data2, text): 

        user_context = text
        
        okt = Okt()
        user_interest_keywords = okt.nouns(user_context)
        
        user_tfidf_vector_1 = tfidf_vectorizer_1.transform([' '.join(user_interest_keywords)])
        user_count_vector_1 = count_vectorizer_1.transform([' '.join(user_interest_keywords)])
        user_ppmi_vector_1 = ppmi_matrix(user_count_vector_1)
        user_nmf_vector_1 = nmf_model_1.transform(user_ppmi_vector_1)

        user_tfidf_vector_2 = tfidf_vectorizer_2.transform([' '.join(user_interest_keywords)])
        user_count_vector_2 = count_vectorizer_2.transform([' '.join(user_interest_keywords)])
        user_ppmi_vector_2 = ppmi_matrix(user_count_vector_2)
        user_nmf_vector_2 = nmf_model_2.transform(user_ppmi_vector_2)

        
        user_lda_vector_1 = lda_model_1.transform(user_count_vector_1)
        user_lda_vector_2 = lda_model_2.transform(user_count_vector_2)
        
        tfidf_cosine_similarities_1 = cosine_similarity(user_tfidf_vector_1, tfidf_matrix_1)
        lda_cosine_similarities_1 = cosine_similarity(user_lda_vector_1, lda_matrix_1)
        nmf_cosine_similarities_1 = cosine_similarity(user_nmf_vector_1, nmf_matrix_1)
        
        tfidf_cosine_similarities_2 = cosine_similarity(user_tfidf_vector_2, tfidf_matrix_2)
        lda_cosine_similarities_2 = cosine_similarity(user_lda_vector_2, lda_matrix_2)
        nmf_cosine_similarities_2 = cosine_similarity(user_nmf_vector_2, nmf_matrix_2)

        data1['time_weight'] = data1['발행(출시)일자'].apply(calculate_time_weight)
        data2['time_weight'] = data2['발행(출시)일자'].apply(calculate_time_weight)


        tfidf_cosine_similarities_weighted_1 = tfidf_cosine_similarities_1 * data1['time_weight'].values[:, np.newaxis]
        lda_cosine_similarities_weighted_1 = lda_cosine_similarities_1 * data1['time_weight'].values[:, np.newaxis]
        nmf_cosine_similarities_weighted_1 = nmf_cosine_similarities_1 * data1['time_weight'].values[:, np.newaxis]

        tfidf_cosine_similarities_weighted_2 = tfidf_cosine_similarities_2 * data2['time_weight'].values[:, np.newaxis]
        lda_cosine_similarities_weighted_2 = lda_cosine_similarities_2 * data2['time_weight'].values[:, np.newaxis]
        nmf_cosine_similarities_weighted_2 = nmf_cosine_similarities_2 * data2['time_weight'].values[:, np.newaxis]

        ensemble_similarities_weighted_1 = (nmf_cosine_similarities_weighted_1 + lda_cosine_similarities_weighted_1 + tfidf_cosine_similarities_weighted_1) / 3
        ensemble_similarities_weighted_2 = (nmf_cosine_similarities_weighted_2 + lda_cosine_similarities_weighted_2 + tfidf_cosine_similarities_weighted_2) / 3
        
        top_indices_1 = ensemble_similarities_weighted_1.argsort()[0][-3:][::-1]
        recommendations_1 = data1.iloc[top_indices_1]
        recommendations_1 = recommendations_1['상품명']

        titles_1 = recommendations_1.values
        titles_1 = titles_1.tolist()

        top_indices_2 = ensemble_similarities_weighted_2.argsort()[0][-1:][::-1]
        recommendations_2 = data2.iloc[top_indices_2]
        recommendations_2 = recommendations_2['상품명']

        titles_2 = recommendations_2.values
        titles_2 = titles_2.tolist()

        titles = titles_1 + titles_2
        return titles
    
    def generate_summary(raw_text):

        llm = OpenAI(temperature=0)

        summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
        summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
        summary =summarize_document_chain.run(raw_text)

        return summary
    
    def translate(raw_text):
        api_key = "b14dfe91-4e6b-1864-840b-4c64eb6149e8"
        translator = deepl.Translator(api_key)
        result = translator.translate_text(raw_text, target_lang="KO")

        return result


    st.title("데이터사이언스 도서 추천 시스템")

    if 'recommendations' not in st.session_state:
        st.session_state['recommendations'] = []
    
    if 'selected_product' not in st.session_state:
        st.session_state['selected_product'] = None

    # 사용자 입력 받기
    user_input = st.text_input("상품 추천을 위한 키워드를 입력하세요:")

    if user_input:
        st.session_state['recommendations'] = generate_recommendations(data1, data2, user_input)
        if not hasattr(st.session_state, 'task'):
            st.session_state.task = st.session_state['recommendations']
            st.write("추천 상품 목록:")
        for product in st.session_state.task:
            st.write(product)

    # 수평선 추가
    st.markdown("---")

    if st.session_state['recommendations']:
        st.session_state['selected_product'] = st.selectbox("요약을 보고 싶은 상품을 선택하세요:", st.session_state.task)

        # rule-base기반으로 한글 3권 원서 1권 추천
        if st.session_state['selected_product'] in data1["상품명"].to_list():
            book_info = data1[data1['상품명'] == st.session_state['selected_product']]['book_info'].iloc[0]
            summary = generate_summary(book_info)
            summary = translate(summary)
            st.write("상품 요약 :", summary)

            product_df = data1[data1['상품명'] == st.session_state['selected_product']][["상품명", "판매가", "인물", "출판사", "분야", "review_rate", "page"]]
            st.dataframe(product_df)

        elif st.session_state['selected_product'] in data2["상품명"].to_list():
            book_info = data2[data2['상품명'] == st.session_state['selected_product']]['index'].iloc[0]
            summary = generate_summary(book_info)
            summary = translate(summary)
            st.write("상품 요약 :", summary)

            product_df = data2[data2['상품명'] == st.session_state['selected_product']][["상품명", "판매가", "인물", "출판사", "분야",  "review_rate", "page"]]
            st.dataframe(product_df)


if __name__ == "__main__":
    main()