
# Python: Differential Privacy study, Laplace distribution

Differentially private queries make use of a class of data transformation algorithms that carry with them a guarantee of privacy loss overall queries, and per query. Namely epsilon differentially private algorithms, as coined in the seminal work by Dwork, et al [Calibrating Noise to Sensitivity in Private Data Analysis](https://link.springer.com/content/pdf/10.1007/11681878_14.pdf).

This case study explores the underlying distribution from which noise is sampled during the course of running a differentially private query (hint: it's the name of the script) on some study data (./data/*.csv). This makes use of [OpenMined's Differential privacy libary](https://github.com/OpenMined/PyDP), which is a python package written using Google's open source C++ [DP library](https://github.com/google/differential-privacy).


## Laplace.py

### Outline
Methods to repeatedly run private statistic generation. Arguments: number of iterations, data (list), privacy budget (epsilon)
* repeated_sum
* repeated_average
* repeated_max

Class to import data from csv, and clean the data
Class methods to generate nonprivate statistic (mean, max, sum)

Code in _main_
* initalize class of clean data from csv
* makes use of repeated_averaage and class method prepareData.nonprivate_average
* generates plots for the study




### Ouput

Running laplace.py as written generates a set of 4 figures:

* annual_income_distribution_plots.png
  * histograms and boxplots of the unmodified source data, demonstrating significant right-skew of the data
  * note: a plot in this series is in log scale, so as to fully appreciate the skewness of the data
* repeated_average_historgram.png
  * observations of repeated private queries on the unmodified source data (e.g. private average)
  * note: center of observations fall on true value
* skew_corrected_repeated_average_historgram.png
  * observations of repeated private queries on the skew-corrected source data
* normalized_noise_observations.png
  * Observations for skew-crrected and unmodified (skewed) data
  * observations in this series are normalized with respect to each source data's true metric (e.g. true average)



### Input

Data is provided in the /data/*.csv files located in the project root directory.
This data was sourced from [Kaggle: Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) challenge and carries an [Universal, Creative Commons Public License](https://creativecommons.org/publicdomain/zero/1.0/)

This data is a fictional dataset that presents a unique user id keyed credit scoring source data. For the purpose of this demonstartion the annualized salary for the last month of the year (month = december) is used.


## Software License

[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)
