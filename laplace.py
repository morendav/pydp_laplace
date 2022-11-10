
"""
Introduction to using pydp library in Python

This intro will cover noise sampling from distibution, by demonstrating the shape of repeated 'private query' observations, without accounting for privacy budget.

The expected result should show:
1. the observations are roughly a laplace distribution, centered on the real measurement
2. the laplace distribution from which the noise is sampled is dependent on the distribution of the source data

This script assumes the source data is located relative to the script as it is in the github repo
Note: a few #todo: for time-of-execution improvements





Copyright (c) 2022, d.l.moreno
All rights reserved.

This source code is licensed under the Apache v2 license found in the
LICENSE file in the root directory of this source tree.
"""

# Standard & third party libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import factorize
from pathlib import Path
from seaborn import histplot
from seaborn import boxplot
from random import randint

# python differential privacy libraries
from pydp.algorithms.laplacian import BoundedMean
from pydp.algorithms.laplacian import BoundedSum
from pydp.algorithms.laplacian import Max


def repeated_sum(iterations, privacy_budget, list):
    """
    repeated_sum computes the private sum repeatedly, over n-iterations
    and returns an array of the results

    method arguments:
        iterations = number of times to repeat the private statistic
        privacy_budget = value of epsilon for differential privacy measurement
        list = list of sample to use as the basis for the statistic
    """
    results = np.zeros(iterations)
    upper_bound = max(list)  # set upper bound for private sum algorithm to max(dataframe column value)

    for i in range(iterations):
        private_metric_object = BoundedSum(
            epsilon=privacy_budget,
            delta=0,
            lower_bound=0,
            upper_bound=upper_bound,
            dtype="float",
        )
        results[i] = private_metric_object.quick_result(list)

    return results


def repeated_average(iterations, privacy_budget, list):
    """
    repeated_average computes the private mean repeatedly,
    over n-iterations and returns an array of the results

    method arguments:
        iterations = number of times to repeat the private statistic
        privacy_budget = value of epsilon for differential privacy measurement
        list = list of sample to use as the basis for the statistic
    """
    results = np.zeros(iterations)
    upper_bound = max(list)

    for i in range(iterations):
        private_metric_object = BoundedMean(
            epsilon=privacy_budget,
            delta=0,
            lower_bound=0,
            upper_bound=upper_bound,
            dtype="float",
        )
        results[i] = private_metric_object.quick_result(list)


    return results


def repeated_max(iterations, privacy_budget, list):
    """
    repeated_max computes the private max repeatedly,
    over n-iterations and returns an array of the results

    method arguments:
        iterations = number of times to repeat the private statistic
        privacy_budget = value of epsilon for differential privacy measurement
        list = list of sample to use as the basis for the statistic
    """
    results = np.zeros(iterations)
    upper_bound = max(list)

    for i in range(iterations):
        private_metric_object = Max(
            epsilon=privacy_budget,
            delta=0,
            lower_bound=0,
            upper_bound=upper_bound,
            dtype="float",
        )
        results[i] = private_metric_object.quick_result(list)

    return results

class prepareData:
    """
    Import data from CSV into a dataframe and clean it up whenever the class is initialized.
    Requires arguments for source data, columns needed, and column data types.
    Columns needed is a list of strings, which map to the columns of CSV that are to be kept.
    Column data types is a dictionary which is used to assert data types in columns kept, replacing non-matching values with NaN in the dataframe

    Also provides methods for data transformation:
    apply_column_filter: will filter out rows where Column is not Filter (e.g. Filter out where Month is not Decemeber)
    clean_skewness: low-pass filter for values in dataframe column, where value is less than Quantile, where Quantile = [0:1]

    Other methods are simply callable statistics queries for the dataframe, e.g. max, mean, etc
    """

    # method to read the data files into a dataframe
    def __init__(self, data_file_location, import_columns, col_dtypes):
        self.data_location = data_file_location
        self.columns = import_columns
        self.column_datatypes = col_dtypes
        self.dataframe = read_csv(self.data_location, header=0, sep=",", usecols=data_columns_needed)
        # clean dataframe
        # strip invalid characters, replace known corruptions
        self.dataframe = self.dataframe.applymap(
            lambda x: str(x).strip('_ ,"') if x is not np.NaN and isinstance(x, str) else x). \
            replace([' and', '', 'nan', '!@9#%8', '#F%$D@*&8'], np.NaN)
        # assert datatypes from column_dtype dictionary
        self.dataframe = self.dataframe.astype(self.column_datatypes)
        # handles multiple column filters, reassign dataframe to drop any column where value is not in filter dictionary
        # assumes each column has a single target value (does not handle where col in [value1, value2, ...]

    # class method to create data slice
    # replaces dataframe, assumes the filter dict has 1:1 key value pairing.
    # TODO: Does not handle where column in [values]
    # TODO: as implmented the order of operations matters, this is currently noncommutative
    def apply_column_filter(self, filter):
        for column_key in filter:
            self.dataframe = self.dataframe[
                self.dataframe[column_key] == filter[column_key]
            ]

    # class method to correct skewness
    # Assumes data is skewed right
    # clips values over quantile and redistributes that assignment randomly within the distribution that is not clipped
    # example: if quantile = 0.9, then a value in 0.99 quantile will be redistributed randomly within 0:0.9 of distribution
    # this redistribution will ensure that count(unique users) does not change
    def clean_skewness(self, column, quantile):
        clip_value = self.dataframe[column].quantile(quantile)
        min_value_in_distribution = self.dataframe[column].min()

        # TODO: this for loop could be optimized, for now ensure quantile is within some reasonable range (e.g. >85%)
        for index_over_clip in self.dataframe.index[self.dataframe[column] > clip_value]:
            self.dataframe.at[index_over_clip, column] = randint(
                round(min_value_in_distribution),
                round(clip_value)
            )

    # class method returns sum of column using nonprivate calculation
    def nonprivate_sum(self, column) -> float:
        return self.dataframe.sum()[column]

    # class method returns average in column using nonprivate calculation
    def nonprivate_average(self, column) -> float:
        return self.dataframe.mean()[column]

    # class method returns maximum value in column using nonprivate calculation
    def nonprivate_max(self, column) -> float:
        return self.dataframe.max()[column]


if __name__ == '__main__':
    # Initialize variables & configure some hardcoded values
    # NOTE: assumes data file directory relative position not changed from git repo
    path = Path(os.path.dirname(os.path.abspath(__file__)))
    # Set of columns to read from sample data
    data_columns_needed = [
        "Customer_ID",
        "Month",
        "Annual_Income",
        "Credit_Utilization_Ratio",
        "Amount_invested_monthly",
        "Monthly_Balance",
    ]
    # Set of columns intended datatypes
    # NOTE: at import most datatypes will be 'object' since the data isn't sanitized
    column_dtype = {
        'Customer_ID': 'object',
        'Month': 'object',
        'Annual_Income': 'float64',
        'Credit_Utilization_Ratio': 'float64',
        'Amount_invested_monthly': 'float64',
        'Monthly_Balance': 'float64'
    }
    # Use dataslice from last month, this dict is used to slice the dataframe for month = december
    column_filter = {'Month': 'December'}
    # iterations are the number of items a single differentially private statistic is calculated
    # does not make use of privacy budget accounting (i.e. privacy budget is constant across all queries)
    iterations = 100
    privacy_budget = 1
    private_histogram_bins = round(iterations * 0.4)

    # Pull in and prep data
    clean_df = prepareData(path / "data/test.csv", data_columns_needed, column_dtype)
    clean_df.apply_column_filter(column_filter)

    # plot and save distribution of data to demonstrate long-tailed distribution
    annual_income_figure, axes = plt.subplots(2, 2)
    annual_income_figure.suptitle('Distribution of Annual Income Showing a Right Skew')
    axes[0,0].set_title('Linear scale distribution')
    axes[0,1].set_title('Log scale distribution')

    # Plot left column: historgram (without scaling) and a boxplot demonstrating extreme right skewness of the data
    annual_income_linear_scale_graph = histplot(
        ax=axes[0,0],
        data=clean_df.dataframe['Annual_Income'],
        shrink=.8,
    ).grid(axis='y')
    boxplot(
        factorize(clean_df.dataframe['Annual_Income'])[1],
        ax=axes[1,0],
        showmeans=True,
        orient='h',
        palette="rocket",
    ).grid(axis='x')

    # Log(X) Histogram plot, and a boxplot demonstrating extreme right skewness of the data
    annual_income_log_scale_graph = histplot(
        ax=axes[0,1],
        data=clean_df.dataframe['Annual_Income'],
        log_scale=True,
    )
    boxplot(
        factorize(clean_df.dataframe['Annual_Income'])[1],
        ax=axes[1,1],
        showmeans=True,
        orient='h',
        palette="rocket",
        # log_scale=True,
    ).grid(axis='x')
    axes[1,1].set_xscale("log")
    axes[0,1].set(ylabel=None) # remove y axis label on upper right subplot (otherwise overlaps with upper left plot)
    axes[0, 0].set(xlabel=None) # remove x asix label on upper left subplot (label is implied, redundant)
    plt.savefig(path / 'annual_income_distribution_plots.png')
    plt.close()

    # run repeated query holding static value for privacy budget without accounting, plot count(observed values)
    # plot and save histogram of values
    laplace_cenetered_average = repeated_average(
        iterations,
        privacy_budget,
        list(clean_df.dataframe['Annual_Income'])
    )

    laplace_cenetered_histogram = histplot(data=laplace_cenetered_average)
    # plot a horizontal line at the non-private average to demonstrated center of private queries
    laplace_cenetered_histogram.axvline(
        clean_df.nonprivate_average('Annual_Income')
    )
    laplace_cenetered_histogram.set(title="Repeated DP Query Observations")
    laplace_cenetered_histogram.set(xlabel="Private statistic observation")
    plt.savefig(path / 'repeated_average_historgram.png')
    plt.close()

    # Correct dataframe's skewness
    # Clip values over quantile X, and redistribute those entries over the left-over, available distribution
    # redistribution will ensure count(unique users) doesn't change
    skewed_clean_dataframe = clean_df.dataframe.copy() # before correcting skew save the dataframe
    clean_df.clean_skewness('Annual_Income', 0.985) # correct skew of clean data

    # repeat statitical measurements for skew corrected dataframe
    skew_corrected_laplace_cenetered_average = repeated_average(
        iterations,
        privacy_budget,
        list(clean_df.dataframe['Annual_Income'])
    )

    # plot skew corrected laplace historgram
    skew_corrected_laplace_cenetered_histogram = histplot(data=skew_corrected_laplace_cenetered_average)
    # plot a horizontal line at the non-private average to demonstrated center of private queries
    skew_corrected_laplace_cenetered_histogram.axvline(
        clean_df.nonprivate_average('Annual_Income')
    )

    skew_corrected_laplace_cenetered_histogram.set(title="Repeated DP Query Observations")
    skew_corrected_laplace_cenetered_histogram.set(xlabel="Private statistic observation")
    plt.savefig(path / 'skew_corrected_repeated_average_historgram.png')
    plt.close()

    # Normalize laplace centered repeated query tuples with respect to non-skew corrected and skew corrected averages
    # compare the laplace cnetered histograms
    # demonstrating which dataframe resulted in greater noise injection when computing private stat
    skewed_nonprivate_average = np.full(
        (iterations),
        skewed_clean_dataframe['Annual_Income'].mean()
    ) # init array of value set to skewed and skew corrected dataset nonprivate average
    skew_corrected_nonprivate_average = np.full(
        (iterations),
        clean_df.nonprivate_average('Annual_Income')
    )

    # normalize array of pydp repeated queries by subrating an array of the nonprivate average
    normalized_skewed_laplace_distribution = (laplace_cenetered_average - skewed_nonprivate_average) / skewed_nonprivate_average
    normalized_skew_corrected_laplace_distribution = (skew_corrected_laplace_cenetered_average - skew_corrected_nonprivate_average)/skew_corrected_nonprivate_average

    # create histograms from normalized repeated private query observations
    # these plots are centered on zero, and demonstrate the width of noise as being proportional to the skewness of the data
    normalized_histograms, axes = plt.subplots(2, 2)
    normalized_histograms.suptitle('Normalized noise observations from private queries')
    axes[0, 0].set_title('Skewed source data')
    axes[0, 1].set_title('Skew corrected source data')

    # Plot left column: historgram (without scaling) and a boxplot demonstrating extreme right skewness of the data
    skewed_source_data_histo = histplot(
        ax=axes[0, 0],
        data=normalized_skewed_laplace_distribution,
        shrink=.8,
    ).grid(axis='y')
    boxplot(
        factorize(normalized_skewed_laplace_distribution)[1],
        ax=axes[1, 0],
        showmeans=True,
        orient='h',
        palette="rocket",
    ).grid(axis='x')

    skew_corrected_data_histo = histplot(
        ax=axes[0, 1],
        data=normalized_skew_corrected_laplace_distribution,
        shrink=.8,
    ).grid(axis='y')
    boxplot(
        factorize(normalized_skew_corrected_laplace_distribution)[1],
        ax=axes[1, 1],
        showmeans=True,
        orient='h',
        palette="rocket",
    ).grid(axis='x')
    # modifty axis labels and scaling
    axes[0,1].set(ylabel=None)
    axes[0, 1].sharey(axes[0, 0])

    plt.savefig(path / 'normalized_noise_observations.png')
    plt.close()
