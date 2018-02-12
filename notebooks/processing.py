import pickle
import re
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
from sklearn.externals import joblib


def preprocessor(text):
    """
    Args: text field
    Returns: cleaned text
    """
    text = re.sub('<[^>]*>', ' ', text)
    text = re.sub('&nbsp;', ' ', text)
    text = re.sub('&amp;', ' ', text)
    text = re.sub('&gt;', ' ', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ' ')
    return text
    
def textSequences(documents, vocab_size, maxlen):
    """
    Word embeddings and sequence padding for text data
    Args: documents which will be trained on:
            - vocabulary size: integer
            - embedding size: integer, suggested initial value: 300
            - tokenizer_name: integer
            
            tokenizer will be persisted in the neural_net_models directory
            
    Returns: padded text sequences with embeddings
    """
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(documents)
    sequences = tokenizer.texts_to_sequences(documents)
    data = pad_sequences(sequences, maxlen=maxlen)
    return data

def tokenizeText(data, num_words):
    """
    Bag of words tokenizer
    Args: training features and number of words
    Returns: bag of words matrix
    """
    tokenize = Tokenizer(num_words=num_words)
    tokenize.fit_on_texts(data)
    joblib.dump(tokenize, 'encoders/tokenize.pkl')
    X = tokenize.texts_to_matrix(data)
    return X

def binarizeLabel(labels):
    """
    Args: label column
    Returns: encoded labels
    """
    encoder = preprocessing.LabelBinarizer()
    encoder.fit(labels)
    joblib.dump(encoder, 'encoders/encoder.pkl')
    y = encoder.transform(labels)
    return y

def encodeLabels(labels):
    """
    Args: label column
    Returns: encoded labels
    """
    encoder = LabelEncoder()
    encoder.fit(labels)
    y = encoder.transform(labels)
    num_classes = np.max(y) + 1
    print("num classes: {}".format(num_classes))
    joblib.dump(encoder, 'encoders/encoder.pkl')
    return y

def myEncoder():
    """
    Returns: saved label encoder
    """
    le = joblib.load('encoders/encoder.pkl') 
    return le

def myTokenizer():
    """
    Returns: saved fitted tokenizer
    """
    my_tokenizer = joblib.load('encoders/tokenize.pkl') 
    return my_tokenizer