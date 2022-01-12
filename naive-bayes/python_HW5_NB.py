
# coding: utf-8

# <h2>Project 3: Na&iuml;ve Bayes</h2>

# <blockquote>
#     <center>
#     <img src="nb.png" width="200px" />
#     </center>
#       <p><cite><center>"All models are wrong, but some are useful."<br>
#        -- George E.P. Box
#       </center></cite></p>
# </blockquote>

# <h3>Introduction</h3>
# <!--Aðalbrandr-->
# 
# <p> You recently decided that you want to take the machine learning course at Cornell University. You're super excited to learn ML but are very confused during lectures. You realize that it's because your German professor just throws in German words while explaining concepts and you have no idea what these mean! You could just tell him you don't understand German but you're a proud engineer! There's got to be a better way. Something that doesn't involve communicating because, you know, that's not our forte (just kidding). So, you decide to create a system that detects every time a word is German and translates it for you in the subtitles. In this project, you will just implement the first part of this system using Na&iuml;ve Bayes to predict if a word is German or English. </p>
# <p>
# <strong>P3 Deadlines:</strong> 
# The deadline for this project is on <strong>Oct 7 (11:59 pm EST)</strong>. The late deadline is on <strong>Oct 9</strong>.
# </p>

# <h3> English and German words </h3>
# 
# <p> Take a look at the files <code>english_train.txt</code> and <code>german_train.txt</code>. For example with the unix command <pre>cat german_train.txt</pre> 
# <pre>
# ...
# bibliothek
# aufzuhalten
# maegde
# rupfen
# leer
# merkte
# sucht
# launenhaften
# graeten
# </pre>
# 
# The problem with the current file is that the words are in plain text, which makes it hard for a machine learning algorithm to learn anything useful from them. We therefore need to transform them into some vector format, where each word becomes a vector that represents a point in some high dimensional input space. </p>
# 
# <p>That is exactly what the following Python function <code>language2features</code> does: </p>

# In[1]:


#<GRADED>
import numpy as np
import sys
from cvxpy import *
from matplotlib import pyplot as plt
#</GRADED>

get_ipython().magic('matplotlib inline')


# In[2]:


def feature_extraction_letters(word, B):
    v = np.zeros(B)
    for letter in word:
        v[ord(letter) - 97] += 1
    return v


# In[3]:


def language2features(filename, B=26, LoadFile=True):
    """
    Output:
    X : n feature vectors of dimension B, (nxB)
    """
    if LoadFile:
        with open(filename, 'r') as f:
            words = [x.rstrip() for x in f.readlines() if len(x) > 0]
    else:
        words = filename.split('\n')
    n = len(words)
    X = np.zeros((n, B))
    for i in range(n):
        X[i,:] = feature_extraction_letters(words[i].lower(), B)
    return X


# <p>It reads every word in the given file and converts it into a 26-dimensional feature vector by mapping each letter to a feature. The generated vector is a histogram containing the number of occurrences of each letter in the word.</p> 
# 
# <p>We have provided you with a python function <code>genFeatures</code>, which calls this function, transforms the words into features and loads them into memory. 

# In[4]:


def genFeatures(dimension, language2features, file_german, file_english):
    """
    function [x,y]=genFeatures
    
    This function calls "language2features.py" to convert 
    words into feature vectors and load training data. 
    
    language2features: function that extracts features from language word
    dimension: dimensionality of the features
    
    Output: 
    x: n feature vectors of dimensionality d [n,d]
    y: n labels (-1 = German, +1 = English)
    """
    
    # Load in the data
    Xgerman = language2features(file_german, B=dimension)
    Xenglish = language2features(file_english, B=dimension)
    X = np.concatenate([Xgerman, Xenglish])
    
    # Generate Labels
    Y = np.concatenate([-np.ones(len(Xgerman)), np.ones(len(Xenglish))])
    
    # shuffle data into random order
    ii = np.random.permutation([i for i in range(len(Y))])
    
    return X[ii, :], Y[ii]


# You can call the following command to load features and labels of all German and English words.

# In[5]:


X,Y = genFeatures(26, language2features, "german_train.txt", "english_train.txt")
xTe, yTe = genFeatures(26, language2features, "german_test.txt", "english_test.txt")


# <h3> Multinomial Na&iuml;ve Bayes Classifier </h3>
# 
# <p> The Na&iuml;ve Bayes classifier is a linear classifier based on Bayes Rule. The following questions will ask you to finish these functions in a pre-defined order. <br></p>
# <p>(a) Estimate the class probability P(Y) in 
# <b><code>naivebayesPY</code></b>
# . This should return the probability that a sample in the training set is positive or negative, independent of its features.
# </p>
# 
# 

# In[42]:


#<GRADED>
def naivebayesPY(x,y):
    """
    function [pos,neg] = naivebayesPY(x,y);

    Computation of P(Y)
    Input:
        x : n input vectors of d dimensions (nxd)
        y : n labels (-1 or +1) (nx1)

    Output:
    pos: probability p(y=1)
    neg: probability p(y=-1)
    """
    
    
    ## TODO 1
    pos = np.sum([1 for label in y if label==1])/(len(y))
    ## TODO 1
    neg =  np.sum([1 for label in y if label==-1])/(len(y))
    return pos, neg


#</GRADED>

pos,neg = naivebayesPY(X,Y)
pos,neg


# <p>(b) Estimate the conditional probabilities P(X|Y) <b>(Maximum Likelihood Estimate)</b> without smoothing in 
# <b><code>naivebayesPXY_mle</code></b>
# .  Use a <b>multinomial</b> distribution as model. This will return the probability vectors  for all features given a class label.
# </p> 

# In[126]:


#<GRADED>

def naivebayesPXY_mle(x,y):
    """
    function [posprob,negprob] = naivebayesPXY(x,y);
    
    Computation of P(X|Y) -- Maximum Likelihood Estimate
    Input:
        x : n input vectors of d dimensions (nxd)
        y : n labels (-1 or +1) (nx1)
    
    Output:
    posprob: probability vector of p(x|y=1) (1xd)
    negprob: probability vector of p(x|y=-1) (1xd)
    """
    
    ## TODO 2
    
    n,d = x.shape
    
    posfreqs = np.zeros(d)
    negfreqs = np.zeros(d)
    for i in range(n):
        if y[i] == 1:
            posfreqs += x[i]
        else:
            negfreqs += x[i]
    totalfreqs = posfreqs + negfreqs
    total = np.sum(totalfreqs)
    
    posprob = (posfreqs / posfreqs.sum())
    negprob = (negfreqs / negfreqs.sum())
    
    ## TODO 2
    
    return posprob, negprob
    
    
#</GRADED>

posprob_mle,negprob_mle = naivebayesPXY_mle(X,Y)
posprob_mle.shape, negprob_mle.shape


# In[116]:


print(X)


# <p>(c) Estimate the conditional probabilities P(X|Y) <b>(Smoothing with Laplace estimate)</b> in 
# <b><code>naivebayesPXY_smoothing</code></b>
# .  Use a <b>multinomial</b> distribution as model. This will return the probability vectors  for all features given a class label.
# </p> 

# In[124]:


#<GRADED>
    
def naivebayesPXY_smoothing(x,y,k=1):
    """
    function [posprob,negprob] = naivebayesPXY(x,y);
    
    Computation of P(X|Y) -- Smoothing with Laplace estimate
    Input:
        x : n input vectors of d dimensions (nxd)
        y : n labels (-1 or +1) (nx1)
    
    Output:
    posprob: probability vector of p(x|y=1) (1xd)
    negprob: probability vector of p(x|y=-1) (1xd)
    """
  
    ## TODO 3
    n,d = x.shape
    
    posfreqs = np.zeros(d) + k
    negfreqs = np.zeros(d) + k
    for i in range(n):
        if y[i] == 1:
            posfreqs += x[i]
        else:
            negfreqs += x[i]
    totalfreqs = posfreqs + negfreqs
    total = np.sum(totalfreqs)
    
    posprob = (posfreqs / posfreqs.sum())
    negprob = (negfreqs / negfreqs.sum())
    ## TODO 3
    
    return posprob, negprob
    
    
#</GRADED>

posprob_smoothing,negprob_smoothing = naivebayesPXY_smoothing(X,Y)

posprob_smoothing.shape,negprob_smoothing.shape


# <p>(d) Solve for the log ratio, $\log\left(\frac{P(Y=1 | X = xtest)}{P(Y=-1|X= xtest)}\right)$, using Bayes Rule. Doing this in log space is important to avoid underflow. Make sure to first log the probabilities and then sum them.
#  Implement this in 
# <b><code>naivebayes</code></b>.
# </p>
# 
# 

# In[134]:


#<GRADED>
def naivebayes(x,y,xtest,naivebayesPXY):
    """
    function logratio = naivebayes(x,y);
    
    Computation of log P(Y|X=x1) using Bayes Rule
    Input:
    x : n input vectors of d dimensions (nxd)
    y : n labels (-1 or +1)
    xtest: input vector of d dimensions (1xd)
    naivebayesPXY: input function for getting conditional probabilities (naivebayesPXY_smoothing OR naivebayesPXY_mle)
    
    Output:
    logratio: log (P(Y = 1|X=xtest)/P(Y=-1|X=xtest))
    """
    
    smoothpos, smoothneg = naivebayesPXY(x,y)
    pypos, pyneg = naivebayesPY(x,y)
    
    n,d = x.shape
    
    ## TODO 4
#     logratio = (np.log(smoothpos) + np.log(pypos)) - (np.log(smoothneg) + np.log(pyneg))
    p_pos = (
        np.log(np.math.factorial(int(xtest.sum()))) -
        np.sum([np.log(np.math.factorial(int(freq))) for freq in xtest if freq > 0]) +
        np.sum([np.log(smoothpos[i])*xtest[i] for i in range(d)])
    )
    p_neg = (
        np.log(np.math.factorial(int(xtest.sum()))) -
        np.sum([np.log(np.math.factorial(int(freq))) for freq in xtest if freq > 0]) +
        np.sum([np.log(smoothneg[i])*xtest[i] for i in range(d)])
    )
    logratio = (p_pos + np.log(pypos)) - (p_neg + np.log(pyneg))
    ## TODO 4
    
    return logratio
    
#</GRADED>
p_smoothing = naivebayes(X,Y,X[0,:], naivebayesPXY_smoothing)
p_mle = naivebayes(X,Y,X[0,:], naivebayesPXY_mle)


# In[135]:


p_smoothing


# <p>(e) Naïve Bayes can also be written as a linear classifier. Implement this in 
# <b><code>naivebayesCL</code></b>.

# In[129]:


#<GRADED>

def naivebayesCL(x,y,naivebayesPXY):
    """
    function [w,b]=naivebayesCL(x,y);
    Implementation of a Naive Bayes classifier
    Input:
    x : n input vectors of d dimensions (nxd)
    y : n labels (-1 or +1)
    naivebayesPXY: input function for getting conditional probabilities (naivebayesPXY_smoothing OR naivebayesPXY_mle)

    Output:
    w : weight vector of d dimensions
    b : bias (scalar)
    """
    
    n, d = x.shape
    ## TODO 5
#     print('n,d are ' + str(x.shape))
    
    PXY_pos, PXY_neg = naivebayesPXY(x,y)
#     print(PXY_pos.shape)
    w = np.zeros(d)
    for alpha in range(d):
        theta_alpha_pos, theta_alpha_neg = PXY_pos[alpha], PXY_neg[alpha]
#         print('thetas retrieved')
#         print(x[:,alpha])
        w_alpha = np.log(theta_alpha_pos) - np.log(theta_alpha_neg)
#         w_alpha = np.sum(w_alpha)
#         print(w_alpha)
        w[alpha] = w_alpha
        
    p_1, p_neg1 = naivebayesPY(x,y)
    b = np.sum(np.log(p_1) - np.log(p_neg1)) 
#     print(b)
    ## TODO 5
    
    return w, b


#</GRADED>

w_smoothing,b_smoothing = naivebayesCL(X,Y, naivebayesPXY_smoothing)
w_mle,b_mle = naivebayesCL(X,Y, naivebayesPXY_mle)


# <p>(f) Implement 
# <b><code>classifyLinear</code></b>
#  that applies a linear weight vector and bias to a set of input vectors and outputs their predictions.  (You can use your answer from the previous project.)
#  
#  

# In[106]:


#<GRADED>

def classifyLinear(x,w,b=0):
    """
    function preds=classifyLinear(x,w,b);
    
    Make predictions with a linear classifier
    Input:
    x : n input vectors of d dimensions (nxd)
    w : weight vector of d dimensions
    b : bias (optional)
    
    Output:
    preds: predictions
    """
    
    n,d = x.shape
    
    ## TODO 6
    preds = np.zeros(n)
    for i in range(n):
        preds[i] = 1 if w.dot(x[i]) + b >= 0 else -1
    ## TODO 6
    
    return preds

#</GRADED>

print('Training error (Smoothing with Laplace estimate): %.2f%%' % (100 *(classifyLinear(X, w_smoothing, b_smoothing) != Y).mean()))
print('Training error (Maximum Likelihood Estimate): %.2f%%' % (100 *(classifyLinear(X, w_mle, b_mle) != Y).mean()))
print('Test error (Smoothing with Laplace estimate): %.2f%%' % (100 *(classifyLinear(xTe, w_smoothing, b_smoothing) != yTe).mean()))
print('Test error (Maximum Likelihood Estimate): %.2f%%' % (100 *(classifyLinear(xTe, w_mle, b_mle) != yTe).mean()))


# You can now test your code with the following interactive word classification script. Use the word 'exit' to stop the program.

# In[54]:


DIMS = 26
print('Loading data ...')
X,Y = genFeatures(DIMS, language2features, "german_train.txt", "english_train.txt")
xTe, yTe = genFeatures(26, language2features, "german_test.txt", "english_test.txt")
print('Training classifier (Smoothing with Laplace estimate) ...')
w,b=naivebayesCL(X,Y,naivebayesPXY_smoothing)
train_error = np.mean(classifyLinear(X,w,b) != Y)
test_error = np.mean(classifyLinear(xTe,w,b) != yTe)
print('Training error (Smoothing with Laplace estimate): %.2f%%' % (100 * train_error))
print('Test error (Smoothing with Laplace estimate): %.2f%%' % (100 * test_error))

yourword = ""
while yourword!="exit":
    yourword = input()
    if len(yourword) < 1:
        break
    xtest = language2features(yourword,B=DIMS,LoadFile=False)
    pred = classifyLinear(xtest,w,b)[0]
    if pred > 0:
        print("%s, is an english word.\n" % yourword)
    else:
        print("%s, is a german word.\n" % yourword)


# <h3> Feature Extraction</h3>
# 
# <p> Similar to how we extracted features from words in <code>feature_extraction_letters</code>, we are going to try another way of doing so. This time, instead of mapping a letter to a feature, we will map a pair of letters to a feature. </p>
#     
# <p>
# Every element in the feature vector will represent a pair of letters (e.g. 'aa', 'ab', 'ac'...) and the element representing the pair of letters that occur in the word will be the number of occurence. Make sure your feature vector <b> ordering is alphabetical </b> i.e. ['aa', 'ab', 'ac'.....'ba', 'bb'......'ca','cb'......]. The length of the feature vector will be $26^2 = 676$ to represent all possible pairs of 26 letters. Assume everything is in lower case.
# </p>
# 
# <p>
# Here's an example, for the word 'mama', elements in the feature vector representing 'ma' will be 2, 'am' will be 1. All the other 674 features will be 0.
# </p>
# 
# <p>
# Please modify <code><b>feature_extraction_letters_pairs</b></code> below to implement this feature extraction.
# </p>

# In[ ]:


#<GRADED>
def feature_extraction_letters_pairs(word, B=676):
    """
    Feature extration from word for pairs
    word: word of the language as a string
    
    Output:
    v : a feature vectors of dimension B=676, (B,)
    """
    v = np.zeros(B)
    
    ## TODO 7
    for i in range(1, len(word)):
        v[(ord(word[i-1])-97)*26 + (ord(word[i])-97)] += 1
    ## TODO 7
    
    return v
    
def language2features_pairs(filename, B=676, LoadFile=True):
    """
    Output:
    X : n feature vectors of dimension B, (nxB)
    """
    if LoadFile:
        with open(filename, 'r') as f:
            words = [x.rstrip() for x in f.readlines() if len(x) > 0]
    else:
        words = filename.split('\n')
    n = len(words)
    X = np.zeros((n, B))
    for i in range(n):
        X[i,:] = feature_extraction_letters_pairs(words[i].lower(), B)
    return X

#</GRADED>


# You can now test your code with the following interactive word classification script:

# In[ ]:


''' result of the Naive Bayes classifier using pairs of letters as features '''
DIMS = 676
print('Loading data ...')
Xp,Yp = genFeatures(676, language2features_pairs, "german_train.txt", "english_train.txt")
xTe, yTe = genFeatures(676, language2features_pairs, "german_test.txt", "english_test.txt")
print('Training classifier (Smoothing with Laplace estimate) ...')
w,b=naivebayesCL(Xp,Yp,naivebayesPXY_smoothing)
train_error = np.mean(classifyLinear(Xp,w,b) != Yp)
print('Training error (Smoothing with Laplace estimate): %.2f%%' % (100 * train_error))
test_error = np.mean(classifyLinear(xTe,w,b) != yTe)
print('Test error (Smoothing with Laplace estimate): %.2f%%' % (100 * test_error))

yourword = ""
while yourword!="exit":
    yourword = input()
    if len(yourword) < 1:
        break
    xtest = language2features(yourword,B=DIMS,LoadFile=False)
    pred = classifyLinear(xtest,w,b)[0]
    if pred > 0:
        print("%s, is an english word.\n" % yourword)
    else:
        print("%s, is a german word.\n" % yourword)


# Why do you think we are not running the naïve Bayes classifier with the __maximum likelihood estimates__ in the previous cell? If you don’t see why, go ahead and try it out.
