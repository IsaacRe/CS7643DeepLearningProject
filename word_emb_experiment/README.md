#cs7643-project
- small edit


Under train.py, there is a experiment_word_emb() function that defines the parameters for a given experiment, and uses them for all 12 embedding options. It then saves the results (train, val loss and acc for each embedding, as well as test scores). 

if you run train.py it will run an experiment with non-stop word data.
- the data is already prepared in TSVs and .npy files


structure_data.py has the functions I used to prepare the data
- it needs a path to the data in the text file
