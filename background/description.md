# 1 Introduction

*Welcome to an extensive Exploratory Data Analysis for the 5th Makridakis forecasting competitions (M5)!* This notebook will grow over the coming days and weeks into a deep dive of all the relevant aspects of this challenge. Here’s all you need to know to get started:

*Some Background:* the [Makridakis competitions](https://en.wikipedia.org/wiki/Makridakis_Competitions) (or *M-competitions*), organised by forecasting expert [Spyros Makridakis](https://en.wikipedia.org/wiki/Spyros_Makridakis), aim to provide a better understanding and advancement of forecasting methodology by comparing the performance of different methods in solving a well-defined, real-world problem. The first M-competition was held in 1982. The [forth competition (M4)](https://www.sciencedirect.com/science/article/pii/S0169207019301128) ran in 2018 and featured “100,000 time series and 61 forecasting methods” (source in link). According to forecasting researcher and practitioner [Rob Hyndman](https://robjhyndman.com/hyndsight/) the M-competitions “have had an enormous influence on the field of forecasting. They focused attention on what models produced good forecasts, rather than on the mathematical properties of those models”. This empirical approach is very similar to Kaggle’s trade-mark way of having the best machine learning algorithms engage in intense competition on diverse datasets. M5 is the first M-competition to be held on Kaggle.



## The goal:

We have been challenged to **predict sales data** provided by the retail giant [Walmart](https://en.wikipedia.org/wiki/Walmart) **28 days** into the future. This competition will run in 2 tracks: In addition to forecasting the values themselves in the [Forecasting competition](https://www.kaggle.com/c/m5-forecasting-accuracy/), we are simultaneously tasked to **estimate the uncertainty** of our predictions in the [Uncertainty Distribution competition](https://www.kaggle.com/c/m5-forecasting-uncertainty). Both competitions will have the same 28 day forecast horizon.



## The data:

We are working with **42,840 hierarchical time series**. [The data](https://www.kaggle.com/c/m5-forecasting-accuracy/data) were obtained in the 3 US states of California (CA), Texas (TX), and Wisconsin (WI). 

> “Hierarchical” here means that data can be aggregated on different levels: 

- item level, 
- department level, 
- product category level, 
- and state level. 

> The sales information reaches back from :

- Jan 2011 to June 2016. 

> In addition to the sales numbers, we are also given corresponding data on :

- prices, 
- promotions, 
- and holidays. 

！！！Note, that we have been warned that **most of the time series contain zero values.**

The data comprises **3049** individual products from <u>**3 categories** and **7 departments**, sold in **10 stores** in **3 states**</u>. The hierachical aggregation captures the combinations of these factors. For instance, we can create 1 time series for all sales, 3 time series for all sales per state, and so on. The largest category is sales of all individual 3049 products per 10 stores for 30490 time series.

The training data comes in the shape of 3 separate files:

- `sales_train.csv`: this is our main training data. It has 1 column for each of the 1941 days from 2011-01-29 and 2016-05-22; not including the validation period of 28 days until 2016-06-19. It also includes the IDs for item, department, category, store, and state. The number of rows is 30490 for all combinations of 30490 items and 10 stores.
- `sell_prices.csv`: the store and item IDs together with the sales price of the item as a weekly average.
- `calendar.csv`: dates together with related features like day-of-the week, month, year, and an 3 binary flags for whether the stores in each state allowed purchases with [SNAP food stamps](https://www.benefits.gov/benefit/361) at this date (1) or not (0).



## The metrics:

This competition uses a **Weighted Root Mean Squared Scaled Error** (RMSSE). Extensive details about the metric, scaling, and weighting can be found in the [M5 Participants Guide](https://mofc.unic.ac.cy/m5-competition/).



## Submission File

Each row contains an `id` that is a concatenation of an `item_id` and a `store_id`, which is either `validation` (corresponding to the Public leaderboard), or `evaluation` (corresponding to the Private leaderboard). You are predicting 28 forecast days (`F1-F28`) of items sold for each row. For the `validation` rows, this corresponds to `d_1914 - d_1941`, and for the `evaluation` rows, this corresponds to `d_1942 - d_1969`. (Note: a month before the competition close, the ground truth for the `validation` rows will be provided.)

The files must have a header and should look like the following:

```
id,F1,...F28
HOBBIES_1_001_CA_1_validation,0,...,2
HOBBIES_1_002_CA_1_validation,2,...,11
...
HOBBIES_1_001_CA_1_evaluation,3,...,7
HOBBIES_1_002_CA_1_evaluation,1,...,4
```













