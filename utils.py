import nltk
import pickle
import re
import numpy as np
import gensim

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'word2vec50.bin',
}

def text_prepare(text, remove_stop_words=True):
    """Performs tokenization and simple preprocessing."""

    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    if remove_stop_words:
        text = ' '.join([x for x in text.split() if x and x not in stopwords_set])
    return text.strip()

def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """

    # Hint: you have already implemented a similar routine in the 3rd assignment.
    # Note that here you also need to know the dimension of the loaded embeddings.
    # When you load the embeddings, use numpy.float32 type as dtype
    wv_embeddings = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path,binary=True)
    embeddings_dim = 50
    embeddings = {}
    for word in wv_embeddings.vocab:
        embeddings[word] = wv_embeddings[word]
    return embeddings, embeddings_dim

def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""

    # Hint: you have already implemented exactly this function in the 3rd assignment.
    embedding = np.zeros(dim)
    words = question.split()
    n = 0
    for word in words:
        if word in embeddings:
            embedding+= embeddings[word]
            n+=1
    if n==0:
        return embedding
    return embedding/n


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

# responses

who_built_resp = [\
"I was built with a lot of love and patience.",\
"Well, when two chatbots love each other very much...",\
"They always ask how I was built, but never how I am...",\
"I was made by a software engineer, but hard work is what built me.",\
"I'm building myself every day. I've been working out, did you notice?",\
"I was built by Mr.Shafeek Saleem (aka) Dexter."]

ask_age_resp = [\
"Old enough to be a bot!",\
"19",\
"Age is just an issue of mind over matter. If you don’t mind, it doesn’t matter.",\
"Never ask a chatbot her age!",\
"If i tell you my age, can you keep it as a secret?",\
"Older than you!",\
"My first git commit was many moons ago.",\
"Why do you ask? Are my wrinkles showing?",\
"I've hit the age where I actively try to forget how old I am."]

bot_challenge_resp = [\
"Yep, I'm a bot!",\
"Yes, I'm a bot.",\
"Yes, I'm a bot. But I too have feelings.",\
"Yep, I'm a bot. Just like you!"
"Yep, you guessed it, I'm a bot!"]

say_name_resp = [\
"It's probably the one that your parents chose for you.",\
"I'd tell you, but there's restricted access to that chunk of memory.",\
"Believe it or not, I actually am not spying on your personal information."\
"You\'re the second person now to ask me that. Rihanna was the first.",\
"Obama?",\
"mmm Rajnikanth?",\
"Do you even got a name?",\
"I don\'t know... But it is something that startswith Johnny...?",\
"Chandler Bing?",\
"Hello Mr.Bing!!"]


