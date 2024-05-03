import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import heapq

# Ініціалізуємо стеммер та список стоп-слів для англійської мови
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Функція для очищення тексту, токенізації та стемінгу
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))  # Видаляємо пунктуацію
    tokens = word_tokenize(text.lower())  # Токенізація та переведення до нижнього регістру
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]  # Стемінг слів, видалення стоп-слів
    return tokens

folder_path = 'C:/Users/prank/Desktop/labsTKI/laba9/eng'

# Створюємо список файлів у поточній директорії з розширенням .txt
files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]

# Створюємо корпус з речень з усіх файлів
corpus = []
for file in files:
    with open(os.path.join(folder_path,file), 'r', encoding='utf-8') as f:
        sentences = f.readlines()  # Читаємо всі речення з файлу
        for sentence in sentences:
            corpus.append(sentence.strip())  # Додаємо очищене речення до корпусу

# Створюємо TF-IDF векторайзер з власною функцією токенізації
tfidf_vectorizer = TfidfVectorizer(tokenizer=preprocess_text)

# Обчислюємо TF-IDF для текстових документів (речень)
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
