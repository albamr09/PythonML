import pandas as pd
import regex as re
import numpy as np
import utils
import os
import scipy.io as io


def processEmail(email_contents, verbose=True):

    vocabList = utils.getVocabList()

    # Init return value
    word_indices = []

    # Lower case
    email_contents = email_contents.lower()
    
    # Strip all HTML
    email_contents =re.compile('<[^<>]+>').sub(' ', email_contents)

    # Handle Numbers
    email_contents = re.compile('[0-9]+').sub(' number ', email_contents)

    # Handle URLS
    email_contents = re.compile('(http|https)://[^\s]*').sub(' httpaddr ', email_contents)

    # Handle Email Addresses
    email_contents = re.compile('[^\s]+@[^\s]+').sub(' emailaddr ', email_contents)
    
    # Handle $ sign
    email_contents = re.compile('[$]+').sub(' dollar ', email_contents)
    
    # get rid of any punctuation
    email_contents = re.split('[ @$/#.-:&*+=\[\]?!(){},''">_<;%\n\r]', email_contents)

    # remove any empty word string
    email_contents = [word for word in email_contents if len(word) > 0]
    
    # Stem the email contents word by word
    stemmer = utils.PorterStemmer()
    processed_email = []
    for word in email_contents:
        # Remove any remaining non alphanumeric characters in word
        word = re.compile('[^a-zA-Z0-9]').sub('', word).strip()
        word = stemmer.stem(word)
        processed_email.append(word)

        if len(word) < 1:
            continue

        try:
            word_indices.append(vocabList.index(word))
        except ValueError:
            pass

    if verbose:
        print('----------------')
        print('Processed email:')
        print('----------------')
        print(' '.join(processed_email))
    return word_indices
    
    
def emailFeatures(word_indices):
    # Total number of words in the dictionary
    n = 1899

    # You need to return the following variables correctly.
    x = np.zeros(n)

    x[word_indices] = 1
    
    return x



"""
---------------------------------------------------------------------------------------------------------------------------------

                                            SPAM Classification
                                            
---------------------------------------------------------------------------------------------------------------------------------
"""

#---------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------- Example Dataset 2 ----------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------


with open(os.path.join('../../../data', 'emailSample1.txt')) as fid:
    file_contents = fid.read()

word_indices  = processEmail(file_contents)

#Print Stats
print('-------------')
print('Word Indices:')
print('-------------')
print(word_indices)

#---------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------- FEATURES --------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------

with open(os.path.join('../../../data/', 'emailSample1.txt')) as fid:
    file_contents = fid.read()

word_indices  = processEmail(file_contents)
features      = emailFeatures(word_indices)

print('\nLength of feature vector: %d' % len(features))
print('Number of non-zero entries: %d' % sum(features > 0))


#---------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------- TRAINING --------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------


data = io.loadmat(os.path.join('../../../data', 'spamTrain.mat'))
X, y= data['X'].astype(float), data['y'][:, 0]

print('Training Linear SVM (Spam Classification)')
print('This may take 1 to 2 minutes ...\n')

C = 0.1
model = utils.svmTrain(X, y, C, utils.linearKernel)


#---------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------- EVALUATION --------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------

p = utils.svmPredict(model, X)

print('Training Accuracy: %.2f' % (np.mean(p == y) * 100))


#---------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------- TEST ----------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------

data = io.loadmat(os.path.join('../../../data', 'spamTest.mat'))
Xtest, ytest = data['Xtest'].astype(float), data['ytest'][:, 0]

print('Evaluating the trained Linear SVM on a test set ...')
p = utils.svmPredict(model, Xtest)

print('Test Accuracy: %.2f' % (np.mean(p == ytest) * 100))


#---------------------------------------------------------------------------------------------------------------------------
#----------------------------------------- INFLUENCE OF CERTAIN WORDS ------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------

idx = np.argsort(model['w'])
top_idx = idx[-15:][::-1]
vocabList = utils.getVocabList()

print('Top predictors of spam:')
print('%-15s %-15s' % ('word', 'weight'))
print('----' + ' '*12 + '------')
for word, w in zip(np.array(vocabList)[top_idx], model['w'][top_idx]):
    print('%-15s %0.2f' % (word, w))