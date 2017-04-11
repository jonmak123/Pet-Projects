In "Job Key Word.py":
 
"compile_jobs" takes an a list of job names as argument and scrap all the search results from those job searches. it returns a pickled dictionary of all the attributes of those jobs.

"word_count_1" takes the dictionary from "compile_jobs" and extracts the unique words (lexicon) and count the occurence {{1 count per job}} and return a dataframe.

"word_ount_2" takes the dictionary from "compile_jobs" and extracts the unique words (lexicon) and count the occurence {{multiple count per job}} and return a dataframe.

"word_cloud" filters the dataframe using a list of key words and return a matplotlib word cloud.

"make_lexicon" returns a list of unique words from the jobs in the dictionary.

"prepare_train_set1" is deprecated.

"prepare_train_set2" takes the dictionary of jobs, count the appearances of unique words with respect to the job and return one-hot formatted vectors with the lexicon and class for machine learning



In "TF.py":

Everything pretty much is standard tensorflow feedforward neural net. The style should resemble that of Sentdex's tutorials'.