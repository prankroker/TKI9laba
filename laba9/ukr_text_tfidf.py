import os
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import heapq

# Функція для токенізації тексту
def tokenize_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in string.punctuation]
    return tokens

# Створюємо список файлів у поточній директорії
files = [file for file in os.listdir() if file.endswith('.txt')]

# Читаємо кожен файл, де кожне речення є окремим документом
corpus = []
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
        for sentence in sentences:
            corpus.append(sentence.strip())

# Створюємо TF-IDF матрицю з усіма словами (без виключення стоп-слів)
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_text, stop_words=None)
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Створюємо словник з ТОП-3 ключових слів для кожного документу (речення)
top_keywords_per_doc = {}
num_top_keywords = 3

for i in range(len(corpus)):
    sentence = corpus[i]
    feature_index = tfidf_matrix[i,:].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
    top_keywords = heapq.nlargest(num_top_keywords, tfidf_scores, key=lambda x: x[1])
    top_keywords_per_doc[sentence] = [feature_names[idx] for idx, _ in top_keywords]

# Виводимо результати для кожного документу (речення)
for sentence, keywords in top_keywords_per_doc.items():
    print(f"Документ: '{sentence}'\nТОП-{num_top_keywords} ключових слів: {keywords}\n")
