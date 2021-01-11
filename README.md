# AI for Trading

This repository contains the projects that I submitted for the Udacity [AI for Trading](https://www.udacity.com/course/ai-for-trading--nd880) nanodegree. Check out the **[Course Syllabus](https://d20vrrgs8k4bvw.cloudfront.net/documents/en-US/AI+for+Trading+Learning+Nanodegree+Program+Syllabus.pdf)** for more detailed descriptions on the projects and curriculum.

## List of Projects
The following is a list of projects along with a brief note about the concepts and tools learnt.

### 1. Trading with Momentum
* Basics of stock markets, stock prices and market mechanics.
* How to calculate _stock returns_ and design a _momentum trading strategy_.
* Perform a statistical test to conclude if there is alpha in the signal based on momentum indicator.

### 2. Breakout Strategy
* Quant workflow for alpha signal generation, alpha combination, portfolio optimization, and trading.
* The _importance of outliers_ and how to detect them. Learn about methods designed to handle outliers.
* Data pre-processing before regression analysis. Linear Regression, statistical tests (eg: `Kolmogorov-Smirnov test `) and commonly-used time series models.
* Stock _volatility_, and how the GARCH model analyses volatility. See how volatility is used in equity trading.
* Learn about _pairs trading_, and study the tools used in identifying stock pairs and making trading decision.

### 3. Smart beta and portfolio optimization

* Smart Beta Optimization Methodology on various financial securities.
* Fundamentals of portfolio risk and return.
* Calculate tracking errors for Portfolio performance analysis.
* Optimize portfolio variance using `cvxpy` to meet desired  criteria and constraints.
* Calculate the turnover of your portfolio and find the best timing to rebalance.
* Find portfolio weights by analyzing fundamental data and by quadratic programming.

### 4. Alpha Research and Factor Modeling
* What are Alpha factors and how to convert factor values into portfolio weights in a dollar neutral portfolio with leverage ratio equals to 1 (i.e., standardize factor values).
* Learn fundamentals of factor models and type of factors. Learn how to compute portfolio variance using risk factor models. Learn time series and cross-sectional risk models.
* Use Quantopian's `zipline` to manage a pipeline of stock universe and generate/compute factor returns.
* Learn how to use PCA to build risk factor models.

### 5. Sentiment Analysis using NLP

NLP pipeline consists of text processing, feature extraction, and modeling.

* Text acquisition from plain text, tabular data, and online resources.
* Simple data cleaning with python `regex` and learn how to apply regular expressions to financial statements such as 10Ks and 10Qs.
* Use `BeautifulSoup` python library to ease the parsing of html and xml files, downloaded using `requests` library.
* `nltk` (_natural language toolkit_) for tokenization, stemming, and lemmatization.
* Quantitatively measure readability of documents using readability indices.
* Convert document into vectors using bag of words and TF-IDF weighting, and metrics to compare similarities between documents.

### 6. Sentiment Analysis with Neural Networks.

* Neural Network Basics - Maximum likelihood, cross entropy, logistic regression, gradient decent, regularization, and practical heuristics for training neural networks.
* Deep Learning with `PyTorch`- build deep neural networks to process
and interpret news data.
* Recurrent Neutral Networks (RNNs)
    - Learn to use RNN to predict simple a Time Series and train a Character-Level LSTM to generate new text based on the text from the book.
    - Learn `Word2Vec` algorithm using the Skip-gram Architecture
* Sentiment Analysis RNN - Implement a recurrent neural network that can predict if the text of a stock twit is positive or negative. Play with different ways of embedding words into vectors.
  - Construct and train LSTM networks for sentiment classification.
  - Run backtests and apply the models to news data for signal generation.

### 7. Combining Signals for Enhanced Alpha
Combine signals on a random forest model for enhanced alpha, while solving the
problem of overlapping samples.
* **Decision Tree** - Learn how to do branching decision tree using entropy and information gain. Implement decision tree using `sklearn` and visualize the decision tree using `graphviz`.

* **Model Testing and Evaluation**: Learn about Type 1 and Type 2 errors, Precision vs. Recall. Cross validation for time series, and using learning curves to determine underfitting and overfitting.

* **Random Forest**: Learn the ensemble random forest method and implement it in sklearn

* **Feature Engineering**: Certain alphas perform better or worse depending on market conditions. Feature engineering creates additional inputs to give models more contexts about the current market condition so that the model can adjust its prediction accordingly.

* **Overlapping Labels**: Mitigate the problem when features are dependent on each other (non-IID).

* **Feature Importance**: Company would prefer simple interpretable models to black-box complex models. Interpretability opens the door for complex models to be readily acceptable. One way to interpret a model is to measure how much each feature contributed to the model prediction (_feature importance_) using `Shapley Additive Explanations`. Learn how sklearn computes features importance for tree-based methods. Learn how to implement `shap tree algorithm` and compute shap values for a single tree model using the `shap` library.

### 8. Backtesting.
* Learn best practices of backtesting and see what overfitting can "look like" in practice.

* Build a fairly realistic backtester that uses the `Barra data`. The backtester will perform
portfolio optimization that includes transaction costs, and you’ll implement it with computational efficiency in mind, to allow for a reasonably fast backtest. This is really helpful when backtesting, because having reasonably shorter runtimes allows you to test and iterate on your alphas more quickly.


## Certificate - [Verification Link](https://confirm.udacity.com/SAJ5SHNY)
![AIT_ND_Certificate](https://user-images.githubusercontent.com/20330371/104146808-0241f480-53f2-11eb-9a5d-f3a8391a4b38.png)
