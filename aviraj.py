
# coding: utf-8

# In[321]:


import time

import numpy as np
import tensorflow as tf

import re
from collections import Counter
from time import sleep

# import utils


# In[322]:


# from urllib.request import urlretrieve
# from os.path import isfile, isdir
# from tqdm import tqdm
# import zipfile

# dataset_folder_path = 'data'
# dataset_filename = 'text8.zip'
# dataset_name = 'Text8 Dataset'

# class DLProgress(tqdm):
#     last_block = 0

#     def hook(self, block_num=1, block_size=1, total_size=None):
#         self.total = total_size
#         self.update((block_num - self.last_block) * block_size)
#         self.last_block = block_num

# if not isfile(dataset_filename):
#     with DLProgress(unit='B', unit_scale=True, miniters=1, desc=dataset_name) as pbar:
#         urlretrieve(
#             'http://mattmahoney.net/dc/text8.zip',
#             dataset_filename,
#             pbar.hook)

# if not isdir(dataset_folder_path):
#     with zipfile.ZipFile(dataset_filename) as zip_ref:
#         zip_ref.extractall(dataset_folder_path)
        
import json

with open('pos_train.json') as json_data:
    d = json.load(json_data)
#       print(d[10][10])

text=[]
l=0

for i in d:
    for j in i:
        for k in j:
            text.append(k)

print (text[:100])


# In[323]:


def preprocess(text):

    # Replace punctuation with tokens so we can use them in our model
    for index, worda in enumerate(text):
#         if (worda=='.' or worda==',' or worda=='"' or worda==';' or worda=='!' or worda=='?' or worda=='(' 
#             or worda==')' or worda=='--' or worda==':' or worda==''):
#             print(line[index])
        if (worda.isalnum()==False):
            text.remove(text[index])
        word_counts = Counter(text)
#     print (word_counts.most_common(3))
    for index,word in enumerate(text):
        if (word_counts[word]< word_counts.most_common(3)[2][1]):
            text[index]='other'
#             print (word)
    return text
#     words = text.split()
    
    
#     text = text.lower()
#     text = text.replace('.', ' <PERIOD> ')
#     text = text.replace(',', ' <COMMA> ')
#     text = text.replace('"', ' <QUOTATION_MARK> ')
#     text = text.replace(';', ' <SEMICOLON> ')
#     text = text.replace('!', ' <EXCLAMATION_MARK> ')
#     text = text.replace('?', ' <QUESTION_MARK> ')
#     text = text.replace('(', ' <LEFT_PAREN> ')
#     text = text.replace(')', ' <RIGHT_PAREN> ')
#     text = text.replace('--', ' <HYPHENS> ')
#     text = text.replace('?', ' <QUESTION_MARK> ')
#     # text = text.replace('\n', ' <NEW_LINE> ')
#     text = text.replace(':', ' <COLON> ')

    
    # Remove all words with  5 or fewer occurences
#     trimmed_words=[]


#     trimmed_words = [word for word in text if word_counts[word] > 5]

#     return trimmed_words

def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: A list where each item is a tuple of (batch of input, batch of target).
    """
    n_batches = int(len(int_text) / (batch_size * seq_length))

    # Drop the last few characters to make only full batches
    xdata = np.array(int_text[: n_batches * batch_size * seq_length])
    ydata = np.array(int_text[1: n_batches * batch_size * seq_length + 1])

    x_batches = np.split(xdata.reshape(batch_size, -1), n_batches, 1)
    y_batches = np.split(ydata.reshape(batch_size, -1), n_batches, 1)

    return list(zip(x_batches, y_batches))


def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: A tuple of dicts.  The first dict....
    """
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab


# In[324]:



# print(text[:100])
text=text[:100]
# words = utils.preprocess(text)
words = preprocess(text)
print(words)


# In[325]:


print("Total words: {}".format(len(words)))
print("Unique words: {}".format(len(set(words))))


# In[326]:


# vocab_to_int, int_to_vocab = utils.create_lookup_tables(words)
vocab_to_int, int_to_vocab = create_lookup_tables(words)
int_words = [vocab_to_int[word] for word in words]


# In[327]:


from collections import Counter
import random

threshold = 1e-5
word_counts = Counter(int_words)
total_count = len(int_words)
freqs = {word: count/total_count for word, count in word_counts.items()}
p_drop = {word: 1 - np.sqrt(threshold/freqs[word]) for word in word_counts}
train_words = [word for word in int_words if random.random() < (1 - p_drop[word])]


# In[328]:


def get_target(words, idx, window_size=5):
    ''' Get a list of words in a window around an index. '''
    
    R = np.random.randint(1, window_size+1)
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = set(words[start:idx] + words[idx+1:stop+1])
    
    return list(target_words)


# In[329]:



def get_batches(words, batch_size, window_size=5):
    ''' Create a generator of word batches as a tuple (inputs, targets) '''
    
    n_batches = len(words)//batch_size
    
    # only full batches
    words = words[:n_batches*batch_size]
    
    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx+batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x]*len(batch_y))
        yield x, y


# In[330]:


train_graph = tf.Graph()
with train_graph.as_default():
    inputs = tf.placeholder(tf.int32, [None], name='inputs')
#     labels = tf.placeholder(tf.int32, [None, None], name='labels')
    labels = tf.placeholder(tf.int32, [None, None], name='labels')


# In[331]:



n_vocab = len(int_to_vocab)
n_embedding =  300
with train_graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_vocab, n_embedding), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs) # use tf.nn.embedding_lookup to get the hidden layer output


# In[332]:


# Number of negative labels to sample
n_sampled = 100
with train_graph.as_default():
    softmax_w = tf.Variable(tf.truncated_normal((n_vocab, n_embedding))) # create softmax weight matrix here
    softmax_b = tf.Variable(tf.zeros(n_vocab), name="softmax_bias") # create softmax biases here
    
    # Calculate the loss using negative sampling
    loss = tf.nn.sampled_softmax_loss(
        weights=softmax_w,
        biases=softmax_b,
        labels=labels,
        inputs=embed,
        num_sampled=n_sampled,
        num_classes=n_vocab)
    
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)


# In[333]:


with train_graph.as_default():
    ## From Thushan Ganegedara's implementation
    valid_size = 16 # Random set of words to evaluate similarity on.
    valid_window = 100
    # pick 8 samples from (0,100) and (1000,1100) each ranges. lower id implies more frequent 
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples, 
                               random.sample(range(1000,1000+valid_window), valid_size//2))

    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embedding = embedding / norm
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))


# In[334]:



# If the checkpoints directory doesn't exist:
get_ipython().system('mkdir checkpoints')


# In[335]:


# Creates a graph.
with tf.device('/device:GPU:1'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
epochs = 10
batch_size = 500
window_size = 10

with train_graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=train_graph) as sess:
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs+1):
        batches = get_batches(train_words, batch_size, window_size)
        start = time.time()
        for x, y in batches:
            
            feed = {inputs: x,
                    labels: np.array(y)[:, None]}
            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)
            
            loss += train_loss
            
            if iteration % 100 == 0: 
                end = time.time()
                print("Epoch {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss/100),
                      "{:.4f} sec/batch".format((end-start)/100))
                loss = 0
                start = time.time()
            
            if iteration % 1000 == 0:
                ## From Thushan Ganegedara's implementation
                # note that this is expensive (~20% slowdown if computed every 500 steps)
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = int_to_vocab[valid_examples[i]]
                    top_k = 8 # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = int_to_vocab[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)
            
            iteration += 1
    save_path = saver.save(sess, "checkpoints/text8.ckpt")
    embed_mat = sess.run(normalized_embedding)


# In[352]:


with train_graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=train_graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    embed_mat = sess.run(embedding)

