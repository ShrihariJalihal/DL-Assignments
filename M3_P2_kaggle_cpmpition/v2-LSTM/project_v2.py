import pandas as pd
import numpy as np
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords # to get collection of stopwords
from nltk.tokenize import word_tokenize
import string
import gensim
import matplotlib.pyplot as plt

# layers of the architecture
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Bidirectional

from keras.preprocessing.text import Tokenizer # to encode text to int
from keras.models import Sequential   # the model
from keras.utils import pad_sequences # to do padding or truncating


toxic_questions = pd.read_csv("train_dataset.csv")
print("Is null: ", toxic_questions.isnull().values.any())
print("Shape: ", toxic_questions.shape)
print(toxic_questions.head(15))
toxic_questions.groupby("target").target.count().plot.bar()
print(toxic_questions["question_text"][10])

def preprocess_text(sen):

    sen = re.sub('<.*?>', ' ', sen) # remove html tag

    tokens = word_tokenize(sen)  # tokenizing words

    tokens = [w.lower() for w in tokens]    # lower case

    table = str.maketrans('', '', string.punctuation)  # remove punctuations
    stripped = [w.translate(table) for w in tokens]

    words = [word for word in stripped if word.isalpha()]  # remove non alphabet
    stop_words = set(stopwords.words('english'))

    words = [w for w in words if not w in stop_words]   # remove stop words
    words = [w for w in words if len(w) > 2]  # Ignore words less than 2

    return words

# Store the preprocessed reviews in a new list
question_lines = []
sentences = list(toxic_questions['question_text'])

for sen in sentences:
    # Call the preprocess_text function on each sentence of the review text
    question_lines.append(preprocess_text(sen))


print(len(question_lines))

print(question_lines[3])

y = toxic_questions['target']
y = np.array(list(map(lambda x: 1 if x == 1 else 0, y)))
print(y, "length: ", len(y))


EMBEDDING_DIM = 100

# Train word2vec model after preprocessing the reviews
model = gensim.models.Word2Vec(sentences=question_lines, vector_size=EMBEDDING_DIM, window=5, workers=4, min_count=1)
print(model)
words = list(model.wv.index_to_key)
print('Vocabulary size: %d' % len(words))
# Save model
filename = "toxicQuestion_embedding_word2vec.txt"
model.wv.save_word2vec_format(filename, binary=False)


import os

embeddings_index = {}
f = open(os.path.join('','toxicQuestion_embedding_word2vec.txt'), encoding="utf-8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embeddings_index[word] = coefs
f.close()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(question_lines)
sequences = tokenizer.texts_to_sequences(question_lines)


word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

# Only consider the first  100 words of each movie review
max_length = 100

review_pad = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
sentiment = y

print('Shape of pad tensor:', review_pad.shape)
print('Shape of sentiment tensor', sentiment.shape)

print(word_index)
print(review_pad)

vocab_size = len(word_index) + 1

# Create a weight matrix for words in the training data
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

for word, index in word_index.items():
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
    # If words not found in embedding matrix will be all 0's
    embedding_matrix[index, :] = embedding_vector

print(vocab_size)

EMBEDDING_DIM = 100

# Define Model
model = Sequential()
embedding_layer = Embedding(vocab_size,
                            EMBEDDING_DIM,
                            weights = [embedding_matrix],
                            input_length = max_length,
                            trainable=False)
model.add(embedding_layer)
model.add(LSTM(32, dropout=0.3, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

print('Summary of the built model...')
print(model.summary())

# Try using different optimizers and different optimizer configs
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])




test_split = 0.2

indices = np.arange(review_pad.shape[0])

review_pad = review_pad[indices]
sentiment = sentiment[indices]

num_test_samples = int(test_split * review_pad.shape[0])

X_train_pad = review_pad[:-num_test_samples]
y_train = sentiment[:-num_test_samples]
X_test_pad = review_pad[-num_test_samples:]
y_test = sentiment[-num_test_samples:]

X_train_pad.shape, y_train.shape, X_test_pad.shape, y_test.shape
history = model.fit(X_train_pad, y_train, batch_size=128, verbose=1, epochs=5, validation_split=0.5)


# acc_train = history.history['accuracy']
# acc_val = history.history['val_accuracy']
# epochs = range(1,6)
# plt.plot(epochs, acc_train, 'g', label='Training accuracy')
# plt.plot(epochs, acc_val, 'b', label='validation accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()


print('Testing...')
model.evaluate(X_test_pad, y_test)

# model predictions on the test data
preds = model.predict(X_test_pad)
n = np.random.randint(0, 9999)

# Predictions (set the threshold as 0.5)
if preds[n] > 0.5:
  print('predicted sentiment : positive')
else:
  print('precicted sentiment : negative')

# Original Labels
if (y_test[n] == 1):
  print('correct sentiment : positive')
else:
  print('correct sentiment : negative')

  # Get the text sequences for the preprocessed movie reviews
reviews_list_idx = tokenizer.texts_to_sequences(question_lines)

print(reviews_list_idx[1])

# Function to get the predictions on the movie reviews using LSTM model
def add_score_predictions(data, reviews_list_idx):

  # Pad the sequences of the data
  reviews_list_idx = pad_sequences(reviews_list_idx, maxlen=max_length, padding='post', truncating='post')

  # Get the predictons by using LSTM model
  review_preds = model.predict(reviews_list_idx)

  # Add the predictions to the movie reviews data
  toxic_questions['sentiment score'] = review_preds

  # Set the threshold for the predictions
  pred_sentiment = np.array(list(map(lambda x : 1 if x > 0.5 else 0,review_preds)))

  # Add the sentiment predictions to the movie reviews
  toxic_questions['predicted sentiment'] = pred_sentiment

  return toxic_questions

# Call the above function to get the sentiment score and the predicted sentiment
data = add_score_predictions(toxic_questions, reviews_list_idx)

# Display the data
data[:20]

print(data);