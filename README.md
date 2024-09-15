The base directory: The base directory of the project is ~/eladi/YA-WEI-SEM-6.
inside of it there are a lot of sub-directories. 


Files you might need to work with:
1) utils/consts.py
2) utils/HDD_HDE.py
3) EXP1/run_test.py


here is a short explanation about each of the files:
1) utils/consts.py
This file includes a list of all of the constants and hyper-parameters we use.
More Specifically, here is a list of all of the constants that you might want to change:


* METRIC_BANDS = 'euclidean'
meaning: determines the metric that is used to calculate the distance between the bands.
possible values: 'euclidean' or 'cosine'.


* METRIC_PIXELS = 'euclidean'
meaning: determines the metric that is used to calculate the distance between the pixels.
possible values: 'euclidean' or 'cosine'.


USAGE: if you want to change a constant you can either change it in consts.py directly, or in the python script you are running, for example: 
in run_test.py: 
consts.METRIC_PIXELS = ‘cosine’.



* HIERARCHICAL_METHOD = 'HDD'
meaning: it is a constant we created for you, which you could set to any value you would like. The usage is explained here:
        

USAGE (+ 2) utils/HDD_HDE.py):
it is used in the following manner:
in the file utils/HDD_HDE.py in the function calc_hdd in lines around 307:


if consts.HIERARCHICAL_METHOD == 'HDD':
            hdd_mat = HDD_HDE.run_method(distances)
 else:
# ya wei's plug in method that given distances between #patches returns distances between patches for classification
            pass


The input distances are in the variable “distances” and the result distance matrix should be put in the variable “hdd_mat”.


so, you could add any ifs that you would like in utils/HDD_HDE.py, and then in your script change the HIERARCHICAL_METHOD value in order to check different methods. 


3) EXP1/run_test.py:
This is a script that runs an experiment “reps” times and outputs the mean train and test accuracies.


USAGE:
Maybe the bash script only_one.sh might be helpful (runs a python script as a task) and if you would like to parallelize there is also the bash  script test.sh.
NOTE: Remember to change the script the the bash script (.sh files) run.


Constants in the file that you might want to change:
In run_test.py around lines 45-50 there are some parameters you might want to play with:


# M is the way to calculate the matrix M which is the input to wasserstein. could be 'euclidean' or 'hdd'. because all of our results re with 'euclidean' i didn’t change the script if the value is different- if you want to check different value let us know and we will chnage it so it is possible.
M = 'euclidean'


# the number of reps to run the test for
reps = 10


# the dataset for the test- possible values:
‘paviaU’ for pavia University dataset
‘pavia’ for pavia Center dataset
‘KSC’ for KSC dataset
dataset_name = 'pavia'


# the factor for both x and y axes for the patching.
there are typical values we use of each of the datasets, which can be found here:
validation results
factor = 9

Just to mention- for all of the datasets and corresponding patch sizes in validation results, the wasserstein distance matrix is already computed- So it should run faster without the need to calculate it.
You could verify it uses the pre-computed values if it print the line:
USING SAVED PRECOMPUTED DISTANCES!
Alternatively, if it computes the distances, it would print:
CALCULATING DISTANCES!


The results should appear at the out file in the end. There are a lot of prints throughout the process, sorry about that.
