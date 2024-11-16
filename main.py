import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Загрузка данных и ограничение первых 1000 строк
file_path = "news.json"
df = pd.read_json(file_path, lines=True)

# Ограничиваемся первыми 1000 строками
df = df.head(10000)

# Проверка данных
print("Первые строки данных:")
print(df.head())

# Загрузка модели для создания эмбеддингов
model = SentenceTransformer('all-MiniLM-L6-v2')  # Легковесная модель, подходящая для большинства задач

# Создание эмбеддингов на основе short_description с визуализацией
embeddings = []
for i, description in enumerate(df['short_description']):
    vector = model.encode(description).tolist()
    embeddings.append(vector)

    # Визуализация процесса
    progress = (i + 1) / len(df) * 100
    print(f"\rСоздание эмбеддингов: {progress:.2f}%", end="")

print("\nЭмбеддинги созданы.")

# Добавление эмбеддингов в DataFrame
df['embedding'] = embeddings

# Преобразование списка эмбеддингов в numpy-массив
embedding_array = np.array(embeddings)

# Сохранение данных с векторами в файл для дальнейшего использования
output_file = "news_with_embeddings.json"
df.to_json(output_file, orient="records", lines=True)
print(f"Данные с эмбеддингами сохранены в {output_file}")


# Функция поиска по запросу
def search_similar(query, top_n=5):
    # Преобразование запроса в эмбеддинг
    query_embedding = model.encode(query)

    # Вычисление косинусной схожести
    similarities = cosine_similarity([query_embedding], embedding_array)
    top_indices = np.argsort(similarities[0])[::-1][:top_n]

    # Возвращаем результаты
    results = df.iloc[top_indices]
    return results[['headline', 'short_description', 'link']]


# Бесконечный цикл поиска
print("\nБесконечный поиск по запросам. Введите 'exit' для выхода.")
while True:
    user_query = input("\nВведите поисковый запрос: ")
    if user_query.lower() == 'exit':
        print("Поиск завершён. До свидания!")
        break

    results = search_similar(user_query)
    print("\nРезультаты поиска:")
    print(results.to_string(index=False))
