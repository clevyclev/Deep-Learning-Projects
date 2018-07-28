
# coding: utf-8

# # Recurring Neural Networks with Keras
# 
# ## Sentiment analysis from movie reviews
# 
# This notebook is inspired by the imdb_lstm.py example that ships with Keras. But since I used to run IMDb's engineering department, I couldn't resist!
# 
# It's actually a great example of using RNN's. The data set we're using consists of user-generated movie reviews and classification of whether the user liked the movie or not based on its associated rating.
# 
# More info on the dataset is here:
# 
# https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification
# 
# So we are going to use an RNN to do sentiment analysis on full-text movie reviews!
# 
# Think about how amazing this is. We're going to train an artificial neural network how to "read" movie reviews and guess  whether the author liked the movie or not from them.
# 
# Since understanding written language requires keeping track of all the words in a sentence, we need a recurrent neural network to keep a "memory" of the words that have come before as it "reads" sentences over time.
# 
# In particular, we'll use LSTM (Long Short-Term Memory) cells because we don't really want to "forget" words too quickly - words early on in a sentence can affect the meaning of that sentence significantly.
# 
# Let's start by importing the stuff we need:

# In[1]:


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb


# Now import our training and testing data. We specify that we only care about the 20,000 most popular words in the dataset in order to keep things somewhat managable. The dataset includes 5,000 training reviews and 25,000 testing reviews for some reason.

# In[2]:


print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=20000)


# Let's get a feel for what this data looks like. Let's look at the first training feature, which should represent a written movie review:

# In[3]:


x_train[0]


# That doesn't look like a movie review! But this data set has spared you a lot of trouble - they have already converted words to integer-based indices. The actual letters that make up a word don't really matter as far as our model is concerned, what matters are the words themselves - and our model needs numbers to work with, not letters.
# 
# So just keep in mind that each number in the training features represent some specific word. It's a bummer that we can't just read the reviews in English as a gut check to see if sentiment analysis is really working, though.
# 
# What do the labels look like?

# In[4]:


y_train[0]


# They are just 0 or 1, which indicates whether the reviewer said they liked the movie or not.
# 
# So to recap, we have a bunch of movie reviews that have been converted into vectors of words represented by integers, and a binary sentiment classification to learn from.
# 
# RNN's can blow up quickly, so again to keep things managable on our little PC let's limit the reviews to their first 80 words:

# In[5]:


x_train = sequence.pad_sequences(x_train, maxlen=80)
x_test = sequence.pad_sequences(x_test, maxlen=80)


# Now let's set up our neural network model! Considering how complicated a LSTM recurrent neural network is under the hood, it's really amazing how easy this is to do with Keras.
# 
# We will start with an Embedding layer - this is just a step that converts the input data into dense vectors of fixed size that's better suited for a neural network. You generally see this in conjunction with index-based text data like we have here. The 20,000 indicates the vocabulary size (remember we said we only wanted the top 20,000 words) and 128 is the output dimension of 128 units.
# 
# Next we just have to set up a LSTM layer for the RNN itself. It's that easy. We specify 128 to match the output size of the Embedding layer, and dropout terms to avoid overfitting, which RNN's are particularly prone to.
# 
# Finally we just need to boil it down to a single neuron with a sigmoid activation function to choose our binay sentiment classification of 0 or 1.

# In[6]:


model = Sequential()
model.add(Embedding(20000, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))


# As this is a binary classification problem, we'll use the binary_crossentropy loss function. And the Adam optimizer is usually a good choice (feel free to try others.)

# In[7]:


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# Now we will actually train our model. RNN's, like CNN's, are very resource heavy. Keeping the batch size relatively small is the key to enabling this to run on your PC at all. In the real word of course, you'd be taking advantage of GPU's installed across many computers on a cluster to make this scale a lot better.
# 
# ## Warning
# 
# This will take a very long time to run, even on a fast PC! Don't execute the next block unless you're prepared to tie up your computer for an hour or more.

# In[8]:


model.fit(x_train, y_train,
          batch_size=32,
          epochs=15,
          verbose=2,
          validation_data=(x_test, y_test))


# OK, let's evaluate our model's accuracy:

# In[9]:


score, acc = model.evaluate(x_test, y_test,
                            batch_size=32,
                            verbose=2)
print('Test score:', score)
print('Test accuracy:', acc)


# 81% eh? Not too bad, considering we limited ourselves to just the first 80 words of each review.
# 
# But again - stop and think about what we just made here! A neural network that can "read" reviews and deduce whether the author liked the movie or not based on that text. And it takes the context of each word and its position in the review into account - and setting up the model itself was just a few lines of code! It's pretty incredible what you can do with Keras.
