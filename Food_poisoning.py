import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from datetime import datetime

nltk.download('stopwords')

def preprocess_texts(texts):
    texts = [text.lower() for text in texts]

    texts = [re.sub(r'[^a-zA-Z\s]', '', text) for text in texts]

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    texts = [' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words]) for text in texts]

    return texts


def cluster_texts(texts, num_clusters=3):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)

    return kmeans, X, vectorizer

def get_food_poisoning_cluster(texts, kmeans_model, vectorizer):
    food_poisoning_keywords = ["food poisoning", "sick", "ill", "upset stomach", "nauseous", "unwell"]

    food_poisoning_texts = [text for text in texts if any(keyword in text for keyword in food_poisoning_keywords)]

    if food_poisoning_texts:
        food_poisoning_vectors = vectorizer.transform(food_poisoning_texts)
        food_poisoning_cluster = kmeans_model.predict(food_poisoning_vectors)
        sorted_indices = sorted(range(len(food_poisoning_texts)), key=lambda x: food_poisoning_cluster[x])
        sorted_food_poisoning_texts = [food_poisoning_texts[i] for i in sorted_indices]

        return sorted_food_poisoning_texts, food_poisoning_cluster

    return None, None


def generate_project_brief(food_poisoning_texts, model_inputs):

    project_brief = "Food Poisoning Incident Project Brief:\n"
    project_brief += f"\nDate of Report: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

    if food_poisoning_texts:
        project_brief += "\nSummary of Incident Details:\n"
        for i, text in enumerate(food_poisoning_texts, start=1):
            project_brief += f"{i}. {text}\n"


        project_brief += "\nRecommendations and Proposed Actions:\n"
        project_brief += "- Conduct an immediate investigation into the food poisoning incident at the mentioned restaurant.\n"
        project_brief += "- Collaborate with local health authorities to assess the extent of the issue and coordinate response efforts.\n"
        project_brief += "- Consider temporary closure of the restaurant until the investigation is complete.\n"
        project_brief += "- Communicate with the public, emphasizing the government's commitment to food safety and public health.\n"

    else:
        project_brief += "No specific food poisoning texts detected in the received messages.\n"

    return project_brief

with open("sample_texts.json", "r") as json_file:
    data = json.load(json_file)

texts = data["texts"]

processed_texts = preprocess_texts(texts)

num_clusters = 3  
kmeans_model, X, vectorizer = cluster_texts(processed_texts, num_clusters)

food_poisoning_texts, _ = get_food_poisoning_cluster(processed_texts, kmeans_model, vectorizer)

model_inputs = {} 

project_brief = generate_project_brief(food_poisoning_texts, model_inputs)

print(project_brief)
