# Popularity, Competition, and Ratings: A Look at Hawaii's Businesses

## Introduction

The dataset I'm using for this project is the [Google Local Reviews Dataset](https://mcauleylab.ucsd.edu:8443/public_datasets/gdrive/googlelocal/). This dataset is compiled by the McAuley Lab at UCSD, containing all historical google map reviews for all states in US for up to September 2021. The dataset is split by location, where each location contains a metadata table including information about each individual businesses, and a review table containing each individual review. For Hawaii specifically, the dataset contains **21507 businesses** and about 3.11 millions reviews.

Many interesting questions can be asked of this dataset. For example, does businesses who respond to reviews get a higher rating? Does having a higher proportion of review with pictures indicate a higher rating, as it might reflect that the customer really like the location? Upon some thinking, I decided to zoom out and investigate this interesting question:

> **Do businesses in more popular regions tend to have a higher average rating?**

My theory is that area with higher traffic means higher competition, which would cause worst-performing businesses to die and get replaced by better-performing ones. If this theory is true, this could lead to an increased average rating of the entire area.

I think this would be helpful for anyone who want to pick restaurants off a map. If my theory is true, then going for a busier area would
be a good strategy to follow. As a result, a business owner deciding where to open should also think about the location more carefully.

I work with the business metadata table, which has **21507 rows** before cleaning. The columns relevant to my question are:

| Column | Description |
| --- | --- |
| `gmap_id` | The unique ID for each business, used to identify individual businesses. |
| `num_of_reviews` | The number of reviews the business has. |
| `avg_rating` | The average rating of the business, from 1.0 to 5.0. This is what I run my permutation test on and predict for the model. |
| `address` | The address of the business. I only use it to extract the zip code. |
| `category` | The category of the business, stored as a list since a business can have several. |
| `price` | The price tier of the business (`$` through `$$$$`). |

I also engineer two columns out of the above:

| Column | Description |
| --- | --- |
| `zip_code` | Extracted from `address`, used to group businesses by location. |
| `zip_popularity` | The number of businesses in the same zip code. This is my measure of how "popular" a region is. |

---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

**Dropping duplicate businesses.** `gmap_id` had fewer unique values than rows, which is odd since it's supposed to be the unique identifier for a business. I checked and found that some IDs appear exactly twice, so I compared the two rows for every such ID field by field, including the nested list and dictionary columns. Every duplicated pair turned out to be an exact copy of each other, which mean it's safe to drop all of such duplicate rows since they're exact duplicate. This is most likely an artifact of how the dataset was scraped rather than two genuinely different businesses sharing an ID.

**Extracting the zip code.** There is no zip code column, so I had to pull it out of `address` myself. This took three attempts:

- My first pattern was just `(\d{5})`, which grabbed any five digits anywhere in the string. That produced obviously wrong zips like `15292`, which came from the address `Pele's Kitchen, 152929 Pahoa Village Rd, Pāhoa, HI 96778`, where the street number got captured instead of the zip.
- So I anchored it to the end of the string with `(\d{5})$`. That got 97.9% of non-missing addresses.
- Looking at what was left, some addresses put the zip at the *beginning* with a `〒` symbol in front of it, which is the Japanese postal mark. My final pattern `^〒?(\d{5})|(\d{5})$` handles both, and gets 99.5%.

I checked that the remaining addresses really does not contain a zip code at all, so the third pattern correctly extract all zip-code that is extractable. Businesses with no address (1.48%) or no extractable zip were dropped, since I cannot place them in a region and the whole question is about regions.

**Computing `zip_popularity`.** Once every business has a zip code, I group by zip code and count how many businesses are in each one, then assign that count back to every business in that zip. This is the density measure my question depends on.

**Cleaning `price`.** `price` is stored as repeated currency symbols, so I converted it to a count of characters: `$` becomes 1, `$$$$` becomes 4. This keeps the ordering intact as a single number instead of four separate columns. A small number of rows use `₩` instead of `$`; I treat those tiers the same way, since the tier is what matters and not the currency.

**Converting `category` to a list.** The categories come in as numpy arrays, which one-hot encoding doesn't work with, so I converted them to Python lists. For businesses with no category at all, I invent an `'Empty'` category to indicate the missingness of ANY category. Since each row can have multiple categories, regression cannot infer missingness easily and it should be encoded directly.

Here is the head of the cleaned DataFrame:

| gmap_id                               |   num_of_reviews |   avg_rating |   zip_code |   zip_popularity | category              |   price |
|:--------------------------------------|-----------------:|-------------:|-----------:|-----------------:|:----------------------|--------:|
| 0x7c00456eecad3111:0x8217f9600c51f33  |               18 |          4.4 |      96762 |               85 | ['Restaurant']        |     nan |
| 0x7c00159b5b1b1d25:0x8d2d85d4a758290e |               18 |          4.1 |      96734 |              530 | ['Recreation center'] |     nan |
| 0x7954d376a8b12db3:0xa51dd57e1cc14ca9 |                6 |          5   |      96793 |              470 | ['Food court']        |     nan |
| 0x7954d370921ff6bd:0x3193ba783e26d032 |                8 |          4.8 |      96732 |              618 | ['Coffee shop']       |       1 |
| 0x7c006df045b01715:0xe945c308688e1a46 |                1 |          5   |      96814 |             1294 | ['Ramen restaurant']  |     nan |

*(`price` shows as `nan` for businesses with no price tier. The missingness of this column is investigated in the missingness section below.)*

### Univariate Analysis

<iframe src="assets/zip-popularity-dist.html" width="800" height="500" frameborder="0"></iframe>

Zip popularity spans several orders of magnitude, from 1 business to over 1,400. Most zip codes are small and a few are very large, so the raw counts are heavily right-skewed. This is what we'd expect if the values are spread more evenly across orders of magnitude than across the raw range.

<iframe src="assets/log-zip-popularity-dist.html" width="800" height="500" frameborder="0"></iframe>

After a log transform this look a lot more like an uniform distribution. This means the number of businesses in each zip code is roughly uniformly distributed in their magnitude. This is also a good news for regression, as it means transforming zip popularity with log would ensure each individual bins are represented evenly.

<iframe src="assets/avg-rating-dist.html" width="800" height="500" frameborder="0"></iframe>

The result was a bit surprising. I expected a truncated normal distribution, with values clustering near 5 stars and the left side like a normal tail. Instead, the distribution look very much like a skewed normal distribution with both half of the normal distribution present, centered at 4.5 stars, with a spike at 5.0 and smaller spikes at every whole-number rating. The whole-number rating spike likely come from businesses with few reviews that all happens to have the same rating, where 5 is just the most common one in these scenario. As expected, once we exclude businesses with fewer than 10 reviews, those spike disappear and the average rating look very much like a left-skewed normal distribution.

I still keep those businesses in, for both the testing and the regression. Excluding them would drop 30% of the data and change the question from "what is the average rating in an area" to "what is the average rating among established businesses in an area", which is not what my hypothesis is about. These businesses turns out to matter a lot later on. They are the ones distorting the trendline in the bivariate analysis section, and they are also where the model perform worse in the fairness analysis section.

### Bivariate Analysis

<iframe src="assets/rating-vs-popularity-aggregated.html" width="800" height="500" frameborder="0"></iframe>

This plot is aggregated by zip code, where each dot is one zip code's average rating against its log popularity. The regression line looks flat and there is essentially no meaningful linear relationship. The fitted line has a slope of 0.0063 with an $R^2$ of 0.0017 and a p-value of 0.676. This graph is evidence against my original theory that higher competition in dense area drive higher average rating.

I do noticed that the variance of the graph decrease significantly from left to right. This means while the average remain roughly the same across all zip code area, area with more business have the average cluster closer to the true mean. This make sense considering we are averaging, and the variance of an average decrease as more data points go into it.

The aggregation is also a problem on its own, since it mean each area contribute equal weight to the regression. An area with only 1 business is weighted the same as an area with 1000 business, but an area with 1 business has an average rating far more noisier than those with more businesses. This mean the current approach would amplify the noise in the small area and make the regression less reliable than it seem. It's worth plotting again with a filter to exclude the small area, or plot again without any aggregation at all, to see if the trend look different.

So I filtered to area with at least 10 businesses first. This is very interesting, the trend seemed to have reversed. To confirm it I plotted it one more time unaggregated, with one dot per business.

<iframe src="assets/rating-vs-popularity-unaggregated.html" width="800" height="500" frameborder="0"></iframe>

This is surprising. The trend reversed to negative in both the unaggregated plot and the aggregated plot filtered to area with at least 10 businesses. The original plot was an inaccurate representation of the underlying trend as it was weighing the small area the same weight as the larger area.

This resembles Simpson's paradox, where an aggregated trend is reversed compared to unaggregated trend. This isn't exactly a Simpson's paradox as the group I used is zip code, and the question of whether businesses' average rating is higher in more popular zip code has no answer within individual group. An answer is only defined when we compare across groups, which is exactly what the first plot did.

The mechanism behind why the trend reversed is similar though, and it has to do with how individual businesses' rating is being weighted in the graph. The rating of 1 individual business have a very high variance and is practically a statistical noise. The aggregation only suppress that noise when there is enough datapoint to aggregate, which is also why the variance on the left side of the first plot is so much higher than the right. By weighing them equal, the regression is amplifying noise and drowning out signals. In this particular dataset, the noise happens to cause the trend to reverse, not that something mysterious is happening behind the scene. With a different location, the trend might be the same aggregated or not, even though the variance issue will remain.

Regardless of what the result shows, my hypothesis is definitely not supported by the evidence presented above. I cannot explain why the trend is negative instead of positive as predicted by my theory. I should not change my hypothesis because of this, as that would be cherry picking after seeing the actual trend in the dataset, which is methodologically non-legitimate. I'll use my original hypothesis for the hypothesis test below.

### Interesting Aggregates

The evidence above shows my hypothesis is wrong and there appears to be a negative correlation instead of a positive one. I would like to investigate whether this trend hold true when other variable change. Looking at how the columns correlate with each other, `avg_rating` is correlated with `price` and `log_num_of_reviews` on top of `log_zip_popularity`. Since `price` is missing for the majority of businesses, I'll look into whether the negative trend hold true across different magnitude of number of reviews instead.

Below I grouped businesses by popularity quartile and review quartile, where Q1 is the smallest value and Q4 is the highest, then compute the mean of the average rating in each cell.

| popularity_quartile | Q1 | Q2 | Q3 | Q4 |
| --- | --- | --- | --- | --- |
| **Q1** | 4.38 | 4.38 | 4.38 | 4.43 |
| **Q2** | 4.31 | 4.32 | 4.36 | 4.38 |
| **Q3** | 4.29 | 4.31 | 4.35 | 4.36 |
| **Q4** | 4.25 | 4.28 | 4.30 | 4.32 |

*(rows = zip popularity quartile, columns = number of reviews quartile)*

If we look down each column, the number is always decreasing. This indicate the negative correlation hold across different review quartile bins. As the number of businesses increase in an area, the average rating of businesses go down, which is the opposite of my hypothesis. What's interesting is that if we look across each row, the rating is increasing monotonically (except for the first row where it stays flat from Q1 to Q3 before it increases in Q4). This means there's a positive correlation between number of reviews and average rating, and it holds across all popularity quartile bin as well. This indicate businesses with more reviews typically end up having a higher average rating.

---

## Assessment of Missingness

### MNAR Analysis

I do not believe `price` is MNAR. MNAR means the missingness of the data depends on the data value itself. For example, people may be less willing to pick certain value in a survey as it feel embarrassing to disclose it. However, Google has no reason to not disclose a price tag that they're aware of. And there is no reason to expect why certain price tag are more likely to go missing.

Instead, what the missingness actually depends on is how confident Google is at the value. These price tag are not entered by businesses and are instead inferred by Google via polls and reviews. The more reviews there are, the more datapoint there is for Google to infer pricing. The permutation test below shows the missingness correlate cleanly with num_of_reviews, with restaurants that have a listed price averaging about 295 more reviews than those without. If a business only have very few reviews, then Google simply does not have enough signal to assign a tier confidently, and this happens no matter if that business is actually cheap or expensive. So the missingness depend on an observed column instead of the price value itself, which make `price` MAR and not MNAR.

### Missingness Dependency

`price` is 81% missing, so it is the obvious column to analyze. I ran permutation tests using the absolute difference in means as the test statistic, to investigate whether the values of another column differ significantly between businesses that have a listed price and those that do not.

Since not all categories of business are able to have a price, it is important to run the permutation test only on businesses that could have one. Otherwise the test can produce a misleading conclusion for any column that happens to correlate with whether a business is able to be priced at all. For example, if businesses that cannot have a price also tend to have fewer reviews, then review count will differ between the two groups, and the test will incorrectly attribute that difference to the missingness of price rather than to the structural fact that these businesses can never be priced.

Running the tests across all 20,791 businesses returned $p < 0.001$ for every column examined, **including `latitude` and `longitude`**. There is no reason why `latitude` or `longitude` should be able to explain why a business have or does not have `price`. To remove this confound, I restrict the analysis to businesses in the restaurant category as all restaurants can be priced.

| column | observed_diff | p_value |
| --- | --- | --- |
| `avg_rating` | 0.14 | 0.00 |
| `num_of_reviews` | 295.46 | 0.00 |
| `zip_popularity` | 4.58 | 0.71 |
| `latitude` | 0.02 | 0.42 |
| `longitude` | 0.03 | 0.33 |

<iframe src="assets/missingness-avg-rating.html" width="800" height="500" frameborder="0"></iframe>

**The missingness of `price` does depend on `avg_rating`.** Restaurants with and without a price label differ by 0.14 stars on average, which the graph show far exceed what is normal to observe if there were no dependence ($p < 0.001$). This make `price` more useful for the regression model as it carries real information about `avg_rating`, which is what I want to predict. It also correlate with `num_of_reviews`, which might have something to do with these labels being generated by Google via user pulls, and only businesses with enough review get assigned a price tag.

**The missingness of `price` does not depend on `zip_popularity`.** The observed difference of 4.58 businesses was exceeded in over 70% of permutations, which is largely explainable by chance. How dense an area is does not tell you anything about whether a restaurant there got a price tier.

<iframe src="assets/missingness-review-count.html" width="800" height="500" frameborder="0"></iframe>

Since the missingness of `price` depends on at least one other observed column, `price` is MAR rather than MCAR. This means businesses with a listed price are systematically different from those without. The missingness itself is informative, and should therefore be encoded as a feature to the regression model. Price should not be invented by imputing as there is no reason to expect them to match reality.

Since the price tag is ordinal ($ < $$ < $$$ < $$$$), I keep it as a numerical feature so the model can fit a single coefficient, which usually work better than forcing one-hot encoding on ordinal variable. Missing prices are set to 0 so that the price term contributes nothing for those rows (0 * coefficient = 0), and a separate binary indicator marks which rows were missing. This lets the model learn the effect of missingness on its own without distorting the price coefficient.

---

## Hypothesis Testing

Formalizing my earlier theory into hypotheses:

- **Null hypothesis**: Businesses' average rating is not related to its zip popularity, and any observed differences is purely due to chance.
- **Alternative hypothesis**: Businesses with a higher zip popularity have a higher average rating, and the observed difference is not due to chance.
- **Test statistic**: The mean average rating of businesses with a zip popularity above the median, minus the mean average rating of businesses below the median. This is a directional test statistic using difference in mean, which is the right choice here because my alternative names a direction. I am not just claiming the two groups differ, I am claiming the denser area is *higher*.
- **One-sided test**: This is a right-tailed test. The p-value is the proportion of differences simulated under the null hypothesis that are greater than or equal to the observed difference. A small right-tail area means the observed difference would be unlikely under the null hypothesis, providing evidence against it.
- **Significance level**: $\alpha = 0.05$.

I used a permutation test instead of a parametric test because a parametric test would assume the rating follows a particular distribution, and the univariate analysis above already showed it is left-skewed with spike at whole numbers. Shuffling the group label does not make any assumption about the distribution.

**Results**: the observed difference (high − low) is **−0.0589**, with a right-tail p-value of **1.0000**.

<iframe src="assets/hypothesis-test.html" width="800" height="500" frameborder="0"></iframe>

**Conclusion**: As $p = 1.0000 > \alpha$, there is insufficient evidence to reject the null hypothesis. We cannot conclude that businesses with a higher zip popularity will have a higher average rating.

The observed difference is −0.0589, which is in the opposite direction from the alternative. This is consistent with the exploratory data analysis section. Since I did not know the actual trend before I make up my theory and hypothesis, I cannot change it after seeing the data, which would be cherry picking.

A tail test in the opposite direction (left tail) will give a p-value of $p < 0.001$, which is highly statistically significant. I ran it out of curiosity only, and it is not the test I set out to do. The data does seems to indicate a statistically significant difference exist, just in the opposite direction as my alternative hypothesis expect. This means while my theory is not supported by the evidence, a correlation exist and could be helpful for the regression model below.

One thing to keep in mind is that this permutation test only tells us the observed gap would be very unlikely if the label were assigned randomly. That is not the same as saying density cause the rating to go down. This is also only Hawaii, so with a different location the trend might look different, as I mentioned in the bivariate analysis section.

---

## Framing a Prediction Problem

**Prediction problem**: I want to predict a business's average rating (`avg_rating`) using metadata about the business and the number of reviews it currently has.

**Type**: This is a **regression** problem, as `avg_rating` is a continuous variable ranging from 1.0 to 5.0.

**Response variable**: I chose `avg_rating` because it is the variable my hypothesis was focused on. Although the test did not support my original hypothesis, the exploratory analysis found that zip popularity, review count, and price are all associated with average rating, so I want to see how well these can predict it. I also use `category`, which is another piece of metadata about the business the model can learn from, and is directly related to my original competition theory.

**Evaluation metric**: I use **RMSE** (root mean squared error). RMSE measures the standard deviation of the model's prediction error in the same units as the response variable. An RMSE of 0.5 means predictions are typically off by about half a star, which is directly interpretable.

I choose it over $R^2$ because $R^2$ describes the proportion of variance explained rather than the size of a typical error, which is less directly interpretable than RMSE. I choose it over MAE because squaring penalizes large errors more heavily, which makes it more useful for separating the performance of different models. Since average ratings are tightly clustered around 4.5, a model that only predicts the mean already achieves a low MAE, leaving little room to distinguish a genuinely better model from a trivial one.

**Information known at the time of prediction**: All features I use come from attributes of the business that Google records independently of its specific rating value. None of `num_of_reviews`, `zip_popularity`, `category`, or `price` are derived from `avg_rating` itself, or only exist after `avg_rating` becomes available, so there is no leakage.

The use of `num_of_reviews` might look like it comes "after" the rating, but it doesn't, since they're both aggregation computed from the review at the same temporal moment. One does not exist after the other one exist first, so one does not depend on the other temporally. The model also isn't a temporal forecasting algorithm as it has no temporal data of how the variable changed over time. What the model learned about instead is a snapshot of moment from September 2021. Increasing the input value of `num_of_reviews` by 20 during prediction only tells you what the average rating tends to look like for a business with 20 extra reviews in September 2021, not what that business's average will look like in the future once it gains 20 more reviews.

---

## Baseline Model

My baseline is a **linear regression** in a single `sklearn` Pipeline, trained on an 80/20 train/test split with `random_state=0`.

**Features:**

| Feature | Type | Encoding |
| --- | --- | --- |
| `num_of_reviews` | Quantitative | Passed through untransformed. |
| `zip_popularity` | Quantitative | Passed through untransformed. This is the central variable of the project, so the baseline includes it in its raw form; the final model revisits how it should be scaled. |
| `price_num` | Ordinal | I kept it numerical so the model only needs one coefficient for it, respecting the ordering ($ < $$ < $$$ < $$$$), and set missing values to 0 so the price term drops out entirely for those rows. |
| `price_missing` | Nominal (binary) | 1 when `price` is missing. Since the price term is 0 for those rows, this indicator can absorb the whole effect of missingness without pulling on the price coefficient. The missingness analysis showed price is MAR and carries information, so it is encoded rather than imputed. |
| `top_category` | Nominal | One-hot encoded with `handle_unknown='ignore'`, so categories appearing only in the test set do not cause errors. For the baseline I use only the first category listed, because `OneHotEncoder` does not work with rows that have multiple categories simultaneously. |

**Performance:**

| | RMSE |
| --- | --- |
| Train | 0.4902 |
| Test | 0.5469 |
| Always predict the training mean | 0.5718 |

**Is this model good?** Not really. The performance of the model isn't much better than that of a trivial mean-predicting model, beating it by only about 0.024 stars. This is expected for the baseline as I haven't applied the log transformation or the multi-category encoding. The first is likely the most significant factor since the values are spread evenly in magnitude and not raw range. Training the model using raw value likely distorted the majority of the relationship that exists, causing large residual error. The second one is likely secondary.

The model already seems to be overfitting, with the train RMSE being almost 0.06 lower than the test RMSE. That gap comes from the one-hot encoded category column, and if the category doesn't provide information that is actually helpful, it'll only make the overfitting worse and increase the model variance. The final model addresses both problems.

---

## Final Model

### Features added

**1. `zip_popularity`, log-transformed.** The baseline already included this column in its raw form, where it barely moved the test RMSE at all, which is itself evidence that the raw scale is the wrong one. Zip popularity spans several orders of magnitude (1 to 1,411) and is roughly uniform across magnitudes rather than across the raw range. A linear model on raw counts would treat the gap between 1 and 100 businesses as equivalent to the gap between 1,300 and 1,400, which does not reflect the actual distribution, which change and spread by proportion instead of raw range. The log transform makes a *proportional* change in density correspond to a constant change in the feature, and make the distribution more uniform as shown during the data analysis step. This ensures the model can properly understand the scale of the change and weight different values correctly.

**2. `num_of_reviews`, log-transformed.** Same reasoning as above. Review counts range from 1 to 9,998 and are heavily right-skewed. Doubling from 10 to 20 reviews is a far more meaningful change than going from 5,000 to 5,010. A log scale will make the distribution uniform and make the model better understand the scales of value.

**3. All categories encoded, not just the first.** The baseline used only `top_category`, discarding every other tag a business carries. A business tagged `['Cafe', 'Coffee shop', 'Ice cream shop']` is genuinely all three, and which one Google happens to list first is arbitrary. I wrote a custom `MultiCategoryEncoder` transformer for this, since `OneHotEncoder` cannot consume a column of lists and `MultiLabelBinarizer` does not fit into a `ColumnTransformer`. It sets a 1 for every category a business belongs to, and drops categories appearing in fewer than `min_count` training businesses. The baseline produced 1,615 one-hot columns, many appearing only once or twice, which might be what drove the observed overfitting. Rather than picking that cutoff by hand, I tune it as a second hyperparameter.

Both log-transformed features are then passed through a `StandardScaler`. Ridge penalize large coefficient, so if one feature has a much larger raw scale than the other, the penalty would apply unevenly between them. Scaling them puts both on the same footing.

### Modeling algorithm

**Ridge regression**, chosen over plain linear regression because the baseline model showed clear overfitting (a train-test RMSE gap of 0.057). Ridge is a linear regression designed to handle multicollinearity and overfitting. It works by introducing an extra penalty term that penalizes large coefficients, trying to minimize the sum of squares of the coefficients. This shrinks the many sparse category columns toward zero and reduces variance. It also introduces a regularization hyperparameter `alpha` that I can tune.

### Hyperparameters

I stated both hyperparameters before running any search, and tuned them jointly with `GridSearchCV` using 5-fold cross-validation on the training set only:

- **`alpha`**, the Ridge regularization strength, over `[0.01, 0.1, 1, 10, 100]`. This makes the tradeoff between training and validation performance, which should mitigate the overfitting issue.
- **`min_count`** on the category encoder, over `[1, 3, 5, 10, 25]`. This controls how aggressively rare categories are dropped, trading the extra signal they carry against the sparsity they add. A value of 1 keeps every category, including those appearing in only one business.

The search ran on the training set only, so the test set stayed unseen. `GridSearchCV` then refits the best model on the full training set before evaluating on the test set.

**Selected values: `alpha = 10` and `min_count = 1`**, keeping all 2,127 category features.

The fact that it picked a `min_count` of 1 is surprising, but make sense when thought retroactively. If a category only appeared for one row, then including it only impact that one row since the column is otherwise 0 for all other rows. By picking a `min_count` of 1, the model is trying to memorize every possible details on the dataset. I compared it against a version with `min_count` of 3, and both train and test RMSE improved marginally by ~0.0005. The fact that the testing RMSE also improved is surprising and I cannot explain why. The model is clearly overfitting to noise, but the noise is harmless and sometimes even helpful for the model, besides the model now carries a lot of extra categories that probably will not appear anywhere else. My suspicion is that the test RMSE might of gotten lowered purely by chance because introducing new variable changes all the other coefficient. By chance, the new coefficient might happen to fit to the test set's noise better. A difference of 0.0005 is extremely low that it is likely just noise from random state. For the purpose of predicting truly unseen data outside of this set, I think a higher `min_count` such as a value of 10 would make more sense and likely perform better. This could be a future area of study.

### Performance

| | Baseline RMSE | Final RMSE |
| --- | --- | --- |
| Train | 0.4902 | 0.5079 |
| Test | 0.5469 | **0.5305** |

*Evaluated on the same train/test split (`random_state=0`) as the baseline, so the numbers are directly comparable.*

The final model achieves a test RMSE of **0.5305**, improved from the baseline's **0.5469**, using the identical train-test split so the two numbers are directly comparable. The train-test gap also narrowed from 0.057 to roughly 0.023, indicating that the regularization substantially reduced the overfitting seen in the baseline. The training RMSE actually got *worse*, which is make sense since Ridge is sacrificing training RMSE for the better test RMSE, which is what matter more in real life. This is the tradeoff when moving between overfitting and underfitting in general, and it will happen to anything that deviate away from Ordinary Least Square as OLS is already what mathematically minimize RMSE on training data. It turns out the training RMSE sacrifice is worth it as the testing RMSE reduced significantly.

---

## Fairness Analysis

**Group X**: Businesses with few reviews, defined as fewer than the median number of reviews in the test set (28).

**Group Y**: Businesses with many reviews, defined as 28 or more.

I chose this split because the univariate analysis showed that businesses with few reviews have noisy and often extreme average ratings, including the large spike at exactly 5.0 stars along with the other whole-number rating spikes. If the model handles these businesses poorly, its predictions would be least reliable exactly where a user is most likely to be looking at an unfamiliar business.

**Evaluation metric**: RMSE. This is a regression model, so classification metrics such as precision and recall do not apply, and RMSE is the same metric I used to evaluate the model throughout.

**Null hypothesis**: The model is fair. Its RMSE for businesses with few reviews and businesses with many reviews is roughly the same, and any observed difference is due to random chance.

**Alternative hypothesis**: The model is unfair. Its RMSE for businesses with few reviews is higher than its RMSE for businesses with many reviews.

**Test statistic**: The difference in RMSE between the two groups (few reviews minus many reviews). This is directional, so a right-tailed test is used.

**Significance level**: $\alpha = 0.05$.

I ran this on the test set only, using the already-fitted final model without refitting it. The permutation shuffles the group labels while holding each business's prediction and true rating fixed, so the null distribution reflects only how the RMSE gap varies under random regrouping.

### Results

| Group | RMSE |
| --- | --- |
| Few reviews (< 28) | 0.6727 |
| Many reviews (≥ 28) | 0.3535 |
| **Observed difference** | **0.3191** |

<iframe src="assets/fairness-test.html" width="800" height="500" frameborder="0"></iframe>

**Resulting p-value**: $p < 0.001$. The observed difference was never exceeded in 1,000 permutations.

**Conclusion**: As $p < \alpha$, we reject the null hypothesis.

The model is close to twice as inaccurate for businesses with few reviews as it is for well-reviewed ones. This is consistent with what the univariate analysis found: a business with only a handful of reviews can land on an extreme average such as exactly 5.0 or exactly 3.0, and none of the features available to the model can anticipate which of those it will be. The model instead predicts something close to the population mean for these businesses, which is badly wrong whenever the true rating is extreme.

It is important to make clear of what this does and does not show. It does not show the model is biased against small businesses, in the sense that it systematically under-predicts their rating. What it shows is that the prediction carries much more error. So a prediction from this model should be trusted far less for a new or small business than for an established one, and anything built on it would be better off showing "result not available" than a single number that look just as trustworthy as everything else.
