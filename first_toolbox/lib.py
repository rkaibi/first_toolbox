import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
try:
    nltk.data.find('punkt')
    nltk.data.find('stopwords')
    nltk.data.find('averaged_perceptron_tagger')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def pre_proc(text):
    text = text.lower()
    text = ''.join(char for char in text if not char.isdigit())
    for punc in string.punctuation:
        text = text.replace(punc,'')
    word_tokens = word_tokenize(text)
    clean_text = [w for w in word_tokens if w not in stop_words]
    lemmatized = [lemmatizer.lemmatize(word) for word in clean_text]
    return ' '.join(word for word in lemmatized)

def print_topics(model, vectorizer):
    for idx, topic in enumerate(model.components_):
        print("topic %d:" %(idx))
        print([(vectorizer.get_feature_names()[i],topic[i])
                for i in topic.argsort()[:-10 - 1:-1]])

def make_model(data, n_topics):
    data['clean_text'] = data['text'].apply(pre_proc)
    vectorizer = TfidfVectorizer().fit(data['clean_text'])
    data_vectorized = vectorizer.transform(data['clean_text'])
    lda_model = LatentDirichletAllocation(n_components=n_topics).fit(data_vectorized)

    return lda_model

def make_pred(data, n_topics, example):
    model = make_model(data, n_topics)
    vectorizer = TfidfVectorizer().fit(data['clean_text'])
    example_vectorized = vectorizer.transform(example)

    return model.transform(example_vectorized)
