---
layout: post
title: "Yelp: Predicting Business Success\n Using Early User Reviews"
date: "Febuary 16, 2016"
category : machinelearning
tagline: ""
tags : [john hopkins, data science, capstone, mooc]
---
{% include JB/setup %}

This study, which was originally submitted as part of capstone project of Data Science Specialisation by John Hopkins University at Coursera, analyses the Yelp data to identify early users` rating pattern and how it can be used to predict final outcome of a business. Since the submission was constrained by the course rubric, this report has been revised post submission, to include a comprehensive detail of the analysis carried out. 

Primary focus of the study was on first 35 reviews of businesses for early prediction of likely business success. The early reviewers were classified into three groups, based on percent of times their ratings were off by atleast 2 stars from the final business rating, defined as `Divergence Score` or `DiSco`. While mean rating of first 35 reviews could predict final outcome of a business success with 80% accuracy, it was found that incorporating average rating by groups based on `DiSco` could improve the accuracy to 89%.

## Introduction 

From a business perspective, a positive feedback from customers can be a strong predictor of eventual business success. But, at a more granular level, there is now an increasing interest in identifying segment of customers with contrarian and/or niche tastes. This has been due to surprising findings (see Reference) that strong preferences by this segment of customers, also called `Harbingers of Failure`, may indicate eventual failure of a product. Early endorsement by these contrarians may lull a business into false sense of security, and inspite of good early reviews, eventually, the customer base may dwindle to an extent that just the small group of these customers with niche tastes may remain. 

This project attempted to identify such `Contrarian` within the Yelp data, prefereably early on in the business life cylce. Rating pattern of early reviewers was used to classify users into different segments. Final objective was to ascertain if, based on the analysis of the impact of each segment on the final outcome of a business, accuracy of a predictive model could be enhanced.


## Methods and Data

### Data

The [capstone data](https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/yelp_dataset_challenge_academic_dataset.zip) is a part of Round 6 of [Yelp Dataset Challenge](http://www.yelp.com/dataset_challenge). Data has been provided in five different datasets:

  * user    : dataset with user name, ids and various other attributes of a user
  * review  : collection of all reviews by the users for various businesses 
  * checkin : time and day of the week when service was availed by users for each business.
  * tip     : short reviews left by users for a business
  * business: details regarding each business, in terms of its location, category, average ratings etc.
  
Data sets provided were in `json` format and was loaded in R using *jsonlite* library. Since loading such large files in R was very time consuming, it was prudent to save them as RDS to avoid repeat loading. 

They were finally converted to data frames, by flattening the lists and column names converted to R acceptable formats, as required. In all, there were 366715 users with their 1569264 reviews on 60785 businesses. No reviews were provided for 399 businesses and thus, were disregarded.

### Impute Missing Category

Businesses, as provided in the dataset, were broadly divided into 23 `category`, with further drill down as `categories` enumerating their core business areas. `category/ categories` features were missing for some of them. In order to impute these missing categories, unigrams of keywords with 98.5% sparsity derived from the corpus of names of businesses for each category, using *tm* package, was used. 
Finally, in keeping with the project objective, three datasets namely, business, user and reviews, were merged together to create a comprehensive dataset for further analysis.

### Normalise User Ratings

All reviews posted by any user included ratings assigned to the business by the user, in a scale of 1 to 5. In order to account for unique rating pattern of each user and develop comparable ratings, it is important to normalise these ratings for each user. 

Normalisation of rating by a user was done by subtracting her rating from her average rating `average_stars`. The scale of rating was converted into five categories as follows:

  - strong like scaled as +2 : when difference between `average_stars` and `stars` > 1
  - like scaled as +1        : when difference between `average_stars` and `stars` > 0 but <= 1
  - okay scaled as 0         : when difference between `average_stars` and `stars` == 0
  - dislike scaled as -1     : when difference between `average_stars` and `stars` < 0 but >= -1
  - strong dislike scaled as -2 : when difference between `average_stars` and `stars` < -1

### Normalise Business Ratings

It is interesting to note that number of reviews provided in the review dataset (1569264) do not match the sum of all reviews by all the users  `sum(user$review_count)`  =  11813654 as given in user dataset. This could mean that the average rating calculated for each business may include ratings by users not included in this dataset.
 
Yelp documentation states that average rating of each business has been calculated by taking a mean of all the ratings for the business. As all the reviews were not provided in the given data set, it was required to recalculate the average business ratings for further analysis of data. It was imperative to check if mean ratings calculated for each business from given user data was significantly different from the given mean ratings of businesses, before doing any normalisation on the same.

![Difference in Distribution of Calculated Mean Rating Vs Given Mean Rating of Business]({{ site.baseurl }}/images/rating_diff.jpg)

Hypothesis testing for paired data on calculated and given mean ratings distributions of businesses was carried out and  the distributions were not found to be statistically significantly different. Thereby, normalised ratings from users were used to recalculate the average business ratings for further analysis.

*Laplace's Rule of Succession* was used for calculating the business rating, taking into account number of reviews under each of five categories: -2,-1,0, +1 and +2. Ratings, calculated thus as `Final Average Business Rating`, were again scaled to -2, -1, 0, +1 and +2. All businesses with `Final Average Business Rating` above zero were considered as 'success', while those below zero as 'failure'. Around 38% of businesses had `Final Average Business Rating` below zero.

### Business Rating Pattern

There were 60785 businesses out of which only 24201 businesses had more than 10 reviews. Moreover, for many of these businesses, reviews were not evenly distributed over the business timeline, with some having reviews consistently throughout, while others had very scant reviews, with duration between consequective reviews sometimes spanning weeks or months.

![Review Pattern]({{ site.baseurl }}/images/random_5.jpg)

In order to understand the rating pattern over a period of time, rolling average of business ratings, averaged over a week to account for weekly variations, were calculated and deviations from `Final Average Business Rating` for successful as well as failed businesses was plotted as time series of reviews. In view of sparsity of reviews in more than 50% of business' timeline, only those businesses with atleast 50 reviews were considered for an unbiased plot.

![Mean Rating Divergence]({{ site.baseurl }}/images/p1.jpg)![Mean Rating Variance]({{ site.baseurl }}/images/p2.jpg)

Average divergence of rolling average of ratings seemed to converge after 25th week review. Also, as expected, there appeared to be high variance in ratings from early reviewers, before stabilising after 25th week review. Prediction of final business outcome made by 25th week review could be of tremendous help to a business owner. 

![Fig 6]({{ site.baseurl }}/images/review_weeks.jpg)![Fig 7]({{ site.baseurl }}/images/plot_outcome_class.jpg)

The 25th week review, on average, corresponded to 35 review counts. Also, business outcome calculated using *Laplace's Rule of Succession* from the users' first 35 reviews was approximately 80% accurate as compared to the range of 82 to 84% accuracy from first 50 to 100 reviews. Even a marginal 4% improvement in business outcome prediction at 35th review could surpass the accuracy achievable after 100th reveiw, which would possibly happen much later in a business timeline. Having a predictive model using first 35 reviews may given a business owner an estimation of likely business outcome by, on an average, first 6 months of his business timeline.

![Fig 4]({{ site.baseurl }}/images/rating_rev_1.jpg)![Fig 5]({{ site.baseurl }}/images/rating_rev_2.jpg)

Moreover, distribution of mean rating of businesses after 35th review revealed a huge overlap in the mean ratings of the two business outcomes. Any business with average rating in the overlap zone, at this juncture, cannot be sure where the business is heading! Also, while a significant number of businesses showed early trends commensurate to their final outcome, there were few with early trend juxtaposed to their final success or failure. 

For predictive analysis, as indicated above, all businesses with 50 reviews and more were considered.

![Fig 6]({{ site.baseurl }}/images/duration_50.jpg "Fig 8")

Since, there was a huge disparatity in duration in which different businesses received their 50th review, those within 30 to 50 weeks time period was selected, approximating to a timespan of six months to an year.

### User Rating Pattern

It would be interesting to see how off are people from the `Final Average Business Rating` in their rating of the businesses. A new metric, `Divergence Score` or `DiSco`, was calculated  for users with atleast 30 reveiws. `DiSco` was defined as percentage of times ratings of a user was off from the `Final Average Business Rating` by atleast 2 stars.

![Fig 6]({{ site.baseurl }}/images/av_diff.jpg "Fig 6") ![Fig 7]({{ site.baseurl }}/images/percent.jpg "Fig 7")

An average user's rating appeared to be off by around 1.1 stars from the `Final Average Business Rating` and had an average `DiSco` of approximately 34%, i.e., on average around 34% of the times a user's rating were off by 2 stars or more from the `Final Average Business Rating`.

![Fig 9]({{ site.baseurl }}/images/Rating_Diff_Percent.jpg "Fig 9") 

Users could be categorised based on number of reviews made by them, as,

  - High: >= 30 reveiws, 
  - Medium: <30 & <= 10 and 
  - Low: < 10
  
While users with reviews less than 10 seemed to have extreme percentages of their ratings off by 1 stars from the `Final Average Business Rating`, other two groups on an average fell within the same range of values. In view of this, users with atleast 10 reviews, termed `active users`, were considered for further analysis.

### Predict Business Outcome From Early Reviews

Based on conditions enumerated above, finally, first 35 reviews by 3177 active users for 5660 businesses were selected for predictive analysis. Users were categorised in three groups based on their `Divergence Score` or `DiSco`. Four combination of groups thus created were used for the analysis.

```
library(knitr)
groups = data.frame(Combination = c("30-50","30-60","20-50","20-60"), 
      A = c("< 30%","< 30%","< 20%","< 20%"),
      B = c(">= 30% & <= 50%",">= 30% & <= 60%",">= 20% & <= 50%",">= 20% & <= 60%"),
      C = c("> 50%","> 60%","> 50%","> 60%"))
      
kable(groups, format = "markdown", align = 'c')

```

Ridge Regression was carried out for predicting business success using mean positive rating by different percentage of active users reviews and overall mean rating by all users in first 35 reviews of a business as predictors, with each of the four combinations of A,B and C categorisations listed above.

For each combination, based on different percentages, namely: 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60 and 65%, of reviews by active users in first 35 reviews of these 5660 businesses, 12 data sets were created such that businesses with atleast that many percent of reviews by active users were selected. Randomly selected 20% of such businesses were held out as test set, while ridge regression was used to train the model on the rest, using *glmnet* package. Mean positive rating by users in each of A,B and C categories were taken as predictor along with the overall mean business rating based on first 35 reviews. 100-fold cross validation was used for tuning the parameters.

```
# Summarise user information by calculating number of reviews by each user and 
# percentage of time his/her rating of a business is more than 2 stars away from 
# the Final Mean Business Rating or DiSco.

user.count <- business %>% group_by(user_id) %>%  
  summarise(review_count = length(user_id), 
          rating_off = length(which(rating_diff >= 2))/length(user_id))

# Categorise users in groups A, B and C based on their DiSco
user.count$user_category = ifelse(user.count$rating_off > 0.5, "C",
                                ifelse(user.count$rating_off >= 0.3, "B", "A"))

# Active Users have consiederd to have atleast 10 reveiws in first 35 reveiws.
active.users = user.count[user.count$review_count >= 10, ]

# Calculate percentage of reveiws by active users in first 35 reveiws of a business
active.user.count <- business %>% group_by(business_id) %>%  
   summarise(active_percent = sum(active_user)/length(user_id))

# SAMPLE CALCULATION
# Assume active user percent to be 50
user_percent = 50

# Select all those businesses with active user percent atleast 50 and extract these 
# businesses and reviews by only active users for further analysis
active.users.business = active.user.count[active.user.count$active_percent >= 
                                            user_percent/100, ]

business.model = business[business$business_id %in% active.users.business$business_id,]
business.model = business.model[business.model$active_user == 1,]
business.model$active_user = NULL
business.model = merge(business.model, 
                       active.users[,c("user_id","user_category")],all.x = TRUE)

# Sample random 20% businesses as test data.
set.seed(user_percent)
data_size = length(unique(business.model$business_id))
test.business = sample(unique(business.model$business_id), 
                       size = round(data_size * 0.2 , digits = 0))
test = business.model[business.model$business_id %in% test.business,]

# Select active users in the test set.
test.user <- unique(test$user_id)

# Find all businesses reviewed by the active users in test set and create a training set.
train.business = business.model[!business.model$business_id %in% test.business,]
train.business = unique(train.business[train.business$user_id %in% test.user,]$business_id)
train = business.model[business.model$business_id %in% train.business,]

# Calculate predictors for training and test sets
train.test.model <- TrainingAndTestingSet(train, test)
train <- as.data.frame(train.test.model[1])
test <- as.data.frame(train.test.model[2])

# Create data matrix for ridge regression
x = data.matrix(train[,2:5])
y = train$success

# Carry out ridge regression and plot the result
cv.ridge=cv.glmnet(x,y,alpha=0, family = "binomial", nfolds = 100, standardize=TRUE)
par(mfrow=c(2, 1))
par(mar=c(3, 3, 3, 3))
plot(cv.ridge$glmnet.fit, "norm",   label=TRUE)
plot(cv.ridge$glmnet.fit, "lambda", label=TRUE)

# Parameter coefficients as calculated by ridge regression
coef(cv.ridge, s = "lambda.min")

# Predict on test set using the ridge regression model
PredTest = predict(cv.ridge, s="lambda.min", newx=as.matrix(test[,2:5]), type="response") 
```
![Fig 10]({{ site.baseurl }}/images/ridge.jpg "Fig 10") 

For each test set, three goodness of fit metrics, namely, 

  * accuracy (tp + tn)/(tp + tn + fp + fn), 
  * precision tp/(tp+fp) and 
  * recall tp/(tp+fn) 
are calculated.

## Results 

Although `mean_rating` or business mean rating after 35th review, was found to be fairly good predictor of final business success with 80% accuracy (as stated above), it was observed from models trained on *glmnet* that including mean positive rating of active users from three categories A, B and C improved overall prediction. 

Best result was obtained for 30-50 combination, where users in group A had DiSco <= 30%, B > 30% & <= 50% and C > 50%. 

![Fig 11]({{ site.baseurl }}/images/accuracy.jpg "Fig 19") 

When atleast 50% of ratings in the test data were from active users, the model could categorise businesses correctly with 89% accuracy, could predict 87% of successful businesses correctly (recall) and approximately 90% of businesses predicted as success were actually successful (precision).


## Discussion 

Research cited below, which was a motivation for this project, was based on actual purchases made by customers. This project sought to make a similar assessment from yelp data which has a very strong volunteer bias. 

Difference in rating patterns of users could be exploited in making a better prediction of final business outcome. It was found that while mean rating after 35th review of a business, on an average, could predict final outcome of a business with 80% accuracy, including mean ratings by `DiSco` categorised users as a predictor could propel the accuracy to 89%. This could be a significant find for business owners to help them make a fair assessment of their business early on in it's life cycle.

## References
- <https://marketing.wharton.upenn.edu/mktg/assets/File/Anderson-Eric%202015_02_05_Harbingers.pdf>
- <http://nonesnotes.com/tag/journal-of-marketing-research/>
