In "scraping.py":

1. Run a loop with "compile_jobs" to run a search on Seek.com and get a pickle of dictionary of dictionaries containing search results

2. Run "make_lexicon" to make new items in dictionaries and get a list of unique words

3. Run "prepare_set" to prepare array of data for modelling


In "NN":

4. Run "train_neural_network" to save the model in a sub folder "temp" (which you have to create) as a ckpt file

5. Run "use_neural_network" with an input of a string (descriptive keywords separated with space) to get prediction from the model, called from the ckpt file saved before



----------------------------------------------------



Test set returns ~ 95% for an input of 
['teacher', 'engineer', 'accountant', 'human resources', 'programmer']




----------------------------------------------------




"word count" and "word_cloud" are for some fancy word cloud plotting