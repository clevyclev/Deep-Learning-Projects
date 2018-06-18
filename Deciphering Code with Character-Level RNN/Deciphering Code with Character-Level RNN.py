
# coding: utf-8

# # Deciphering Code with Character-Level RNN
# 
# In this notebook, we'll look at how to build a recurrent neural network and train it to decipher strings encrypted with a certain cipher.
# 
# This exercise will make you familiar with the techniques of preprocessing and model-building that will come in handy when you start building more advanced models for machine translation, text summarization, and beyond.
# 
# ## Dataset
# The dataset we have consists of 10,000 encrypted phrases and the plaintext version of each encrypted phrase.
# 
# Let's start by loading up the dataset to get more familiar with it.

# In[1]:


import helper

codes = helper.load_data('cipher.txt')
plaintext = helper.load_data('plaintext.txt')


# Now `codes` and `plaintext` are both arrays with each element being a phrase. The first three encoded phrases are:

# In[2]:


codes[:5]


# And their plaintext versions are:

# In[3]:


plaintext[:5]


# ## Model Overview: Character-Level RNN
# The model we will use here is a character-level RNN since the cipher seems to work on the characer level. In a machine translation scenario, a word-level RNN is the more common choice.
# 
# A character-level RNN will take as input an integer referring to a specific character and output another integer. To be able to get our model to work, we'll need to preprocess our dataset in the following steps:
#  1. Isolating each character as an array element (instead of an entire phrase, or word being the element of the array)
#  1. Tokenizing the characters so we can turn them from letters to integers and vice-versa
#  1. Padding the strings so that all the inputs and outputs can fit in matrix form
#  
# To visualize this processing, let's assume either our source sequences (`codes` in this case) or target sequences (`plaintext` in this case) look like this (a list of strings):
# 
# <img src="list_1.png" />
# 
# Since this model will be working on the character level, we'll need to separate each string into a list of characters (implicitly done by the tokenizer in this notebook):
# 
# <img src="list_2.png" />
# 
# Then, the process of tokenization will turn each character into an integer.  Note that when you're working on the a word-level RNN (as in most machine translation examples), the tokenizer will assign an integer to each word rather than each letter, and each cell would represent a word rather than a character.
# 
# <img src="list_3.png" />
# 
# Most machine learning platforms expect the input to be a matrix rather than a list of lists. To turn the input into a matrix, we need to find the longest member of the list, and pad all shorter sequences with 0. Assuming 'and two' is the longest sequence in this example, the matrix ends up looking like this:
# 
# <img src="padded_list.png" />
#  
# ## Preprocessing (IMPLEMENT)
# For a neural network to predict on text data, it first has to be turned into data it can understand. Text data like "dog" is a sequence of ASCII character encodings.  Since a neural network is a series of multiplication and addition operations, the input data needs to be number(s).
# 
# We can turn each character into a number or each word into a number.  These are called character and word ids, respectively.  Character ids are used for character level models that generate text predictions for each character.  A word level model uses word ids that generate text predictions for each word.  Word level models tend to learn better.
# 
# Turn each sentence into a sequence of words ids using Keras's [`Tokenizer`](https://keras.io/preprocessing/text/#tokenizer) function. Since we're working on the character level, make sure to set the `char_level` flag to the appropriate value. Then, fit the tokenizer on x.

# In[4]:


from keras.preprocessing.text import Tokenizer


def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    # TODO: Implement
    x_tk = Tokenizer(char_level = True)
    x_tk.fit_on_texts(x)
   

    return x_tk.texts_to_sequences(x), x_tk

# Tokenize Example output
text_sentences = [
    'The quick brown fox jumps over the lazy dog .',
    'By Jove , my quick study of lexicography won a prize .',
    'This is a short sentence .']
text_tokenized, text_tokenizer = tokenize(text_sentences)
print(text_tokenizer.word_index)
print()
for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(sent))
    print('  Output: {}'.format(token_sent))


# ### Padding (IMPLEMENTATION)
# When batching the sequence of word ids together, each sequence needs to be the same length.  Since sentences are dynamic in length, we can add padding to the end of the sequences to make them the same length.
# 
# Make sure all the cipher sequences have the same length and all the plaintext sequences have the same length by adding padding to the **end** of each sequence using Keras's [`pad_sequences`](https://keras.io/preprocessing/sequence/#pad_sequences) function.

# In[7]:


import numpy as np
from keras.preprocessing.sequence import pad_sequences


def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    # TODO: Implement
    # Find the length of the longest string in the dataset. 
    # Then, pass it to pad_sentences as the maxlen parameter
    if length == None:
        longest = 0
        for sequence in x:
            if len(sequence) > longest:
                longest = len(sequence)
        length = longest
    return pad_sequences(x, maxlen=length, padding='post')

# Pad Tokenized output
test_pad = pad(text_tokenized)
for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(np.array(token_sent)))
    print('  Output: {}'.format(pad_sent))


# ### Preprocess Pipeline
# Your focus for this project is to build neural network architecture, so we won't ask you to create a preprocess pipeline.  Instead, we've provided you with the implementation of the `preprocess` function.

# In[8]:


def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_code_sentences, preproc_plaintext_sentences, code_tokenizer, plaintext_tokenizer =    preprocess(codes, plaintext)

print('Data Preprocessed')


# In[9]:


preproc_code_sentences[0]


# In[12]:


from keras.layers import GRU, Input, Dense, TimeDistributed
from keras.models import Model
from keras.layers import Activation
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy


def simple_model(input_shape, output_sequence_length, code_vocab_size, plaintext_vocab_size):
    """
    Build and train a basic RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param code_vocab_size: Number of unique code characters in the dataset
    :param plaintext_vocab_size: Number of unique plaintext characters in the dataset
    :return: Keras model built, but not trained
    """
    # TODO: Build the model
    learning_rate = 1e-3
    
    input_sequence = Input(input_shape[1:])
    rnn = GRU(64,return_sequences=True)(input_sequence)
    logits = TimeDistributed(Dense(plaintext_vocab_size))(rnn)
    
    model = Model(input_sequence, Activation('softmax')(logits))
    model.compile(loss=sparse_categorical_crossentropy,
                 optimizer=Adam(learning_rate),
                 metrics=['accuracy'])
    
    return model


# Reshaping the input to work with a basic RNN
tmp_x = pad(preproc_code_sentences, preproc_plaintext_sentences.shape[1])
tmp_x = tmp_x.reshape((-1, preproc_plaintext_sentences.shape[-2], 1))


# In[13]:


# Train the neural network
simple_rnn_model = simple_model(
    tmp_x.shape,
    preproc_plaintext_sentences.shape[1],
    len(code_tokenizer.word_index)+1,
    len(plaintext_tokenizer.word_index)+1)


# In[14]:


simple_rnn_model.fit(tmp_x, preproc_plaintext_sentences, batch_size=32, epochs=4, validation_split=0.2)


# In[15]:


def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

print('`logits_to_text` function loaded.')

print(logits_to_text(simple_rnn_model.predict(tmp_x[:1])[0], plaintext_tokenizer))


# In[16]:


plaintext[0]


# And there it is. The RNN was able to learn this basic character-level cipher (which was a simple [Caesar cipher](https://en.wikipedia.org/wiki/Caesar_cipher). If you want a bigger cryptography challenge, check out [Learning the Enigma with Recurrent Neural Networks](https://greydanus.github.io/2017/01/07/enigma-rnn/). 
