# BookRecommendation

## 개인화 맞춤 도서 추천 시스템

- LDA 및 다양한 텍스트 마이닝 기법을 통하여 사용자의 입력을 받아 입력 context에 맞는 개인화 맞춤 데이터 사이언스 도서 추천 시스템

---

## 프로세스

![image](https://github.com/Eastha0526/BookRecommendation/assets/110336043/9f4165f4-ea5d-49a6-876c-bba5a736c2f9)


---

### 파일 구조

    Book_recsys/
    ├── data/
    │   ├── corpus.txt
    │   ├── stopword.txt
    │   ├── merge_data_review.csv
    │   └── translate.csv
    ├── EDA.ipynb
    ├── Web Scraping.ipynb
    ├── m.model
    ├── m.vocab
    └── main.py

- data : 말뭉치, 불용어사전, 국내 도서 및 원서 데이터
- m.model, m.vocab : 부분단어 토큰화를 위한 모델과 단어목록
- main.py : 추천 시스템 실행 파일

> CLI : streamlit run main.py

### Library

python : 3.9.13 (Mac OS)

- pandas==1.4.4
- numpy==1.24.4
- scikit-learn==1.3.2
- nltk==3.7
- konlpy==0.6.0
- sentencepiece==0.1.99
- langchain==0.0.339
- deepl==1.16.1
- streamlit==1.28.2

### Reference

[1] 가마자 마사히로, 이즈카 고지로, 마쓰무라 유야 『추천 시스템 입문』, 한빛미디어(2023) </br>
[2] 차루 아가르왈 『추천 시스템』, 에이콘출판(2021) </br>
[3] 임일, 『Python을 이용한 개인화 추천 시스템』, 청람(2022) </br>
[4] https://www.mckinsey.com/industries/retail/our-insights/how-retailers-can-keep-up-with-consumers </br>
[5] Mathew, Praveena, Bincy Kuriakose, and Vinayak Hegde. "Book Recommendation System through content based and collaborative filtering method." 2016 International conference on data mining and advanced computing (SAPIENCE). IEEE, 2016. </br>
[6] Rajpurkar, Sushama, et al. "Book recommendation system." International Journal for Innovative Research in Science & Technology 1.11 (2015): 314-316. </br>
[7] GPT API : https://platform.openai.com/docs/api-reference </br>
[8] DeepL API : https://www.deepl.com/ko/docs-api

