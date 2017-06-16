Project still under consturction.

Objective: Use recurrent neural network to model the news headlines (collected gently from Google News) and the stock price of that day (or the last N days)



Version 1.0 notes
***********************************************

GFinance.py pulls stock prices from google

scrap_Sellenium.py uses a Firefox Sellenium driver to GENTLY scrape Google News

scrap_urllib.py was written but not used as the automated scraping got myself blocked by Google

Processing.py turns the news headlines and stock prices into bag of words and eventually x and y input for the RNN

RNN takes the pickled x and y inputs from Processing.py as batches and run it through a recurrent neural network (basic lstm cell) but seems to be using the wrong learning rate as convergence of test set accuracy does not seem to converge at all (in 5 epochs)

TTD: Give it time and see if convergence happens, also play with the learning rate and hidden layer dimension size
***********************************************