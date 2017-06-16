Still undergoing testing

During the project Stock Sentdex I thought we could apply it on Sentdex's (the youtube channel) sample dataset to do some sentiment analysis on sentences.




Version 1.0
************************************
RNN.py holds everything the model needs. It takes the sample positive and negative statements (files) and turn them into bag of words and one-hot matrices using the NLTK module. The data is fed into the RNN model one mini batch at a time to minimise the computational load. 

Learning rate seems to be quite off as convergence is almost non-existent. Undergoing further testing.
************************************