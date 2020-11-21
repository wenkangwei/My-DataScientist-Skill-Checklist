---
title: DataScientist-Checklist
tags:
  - DataScience
toc: true
categories:
  - Checklist
mathjax: true
comments: true
date: 2020-07-19 13:32:14
---

<!-- more -->
## Checklist
#### Programming
- [ ] Sorting: quicksort (linked list), mergesort (linked list),  insertion, selection, bubble
- [ ] Searching: binary search, binary search tree

**Data Structure**
- [ ] linked List: Search, Insert/append, delete, 
- [ ] **Recursion**
- [ ] Binary Tree/Binary Search Tree/ Completed Tree/ Perfect Tree
- [ ] Balanced Tree
    + [ ] Traversal / Search one or more element
    + [ ] Append node/subtree
    + [ ] Rotate subtree/ Transform subtree
    + [ ] Delete
    + [ ] **Application on Real Problem**
- [ ] Hashtable/ Dictionary
    + [ ] Principle and concept, like hast-collision and its solution
    + [ ] When to use (Time Complexity O(1), Memory complexity O(n), without order)
- [ ] Queue
    + [ ] Basic Append/enqueue and delete/dequeue
    + [ ] Find min, max, search
    + [ ] Application on Real problem / **When to use**
- [ ] Stack
    + [ ] Basic push, pop
    + [ ] find min, max, search
    + [ ] **multiple stacks** for solving real problem
- [ ] Heap
    + [ ] Insertion, Pop
    + [ ] Heapsort
- [ ] Dictionary Tree (Advance): save memory and faster searching
- [ ] Segment Tree

- [ ] Dynamic programming
- [ ] Depth First Search
- [ ] Breath First Search
- [ ] Graph Search

Object-Oriented-Design
- [ ] Class Encapsulation, inheritance, polymorphism

**SQL**
- [ ] Basic Query and Sub-Query Operation
- [ ] Join Operations
- [ ] Aggregation Operations
- [ ] Window Functions

#### Statistic and Machine Learning
- [ ] Possibility and Bayesian Theorem
    + [ ] conditional possibility
    + [ ] total possibility
    + [ ] independence (P(x,y)=P(x)P(y) ) and Conditional independence ( P(x,y|z) = P(x|z)P(y|z) )
- [ ] Normal Distribution
- [ ] Maximum Likelihood Estimation
- [ ] Time Series
    + [ ] IID event and stationary process
    + [ ] Auto-correlation variance function (ACVF) and Auto-Correlation Function (ACF), Partial  Auto-correlation function (PACF)
    + [ ] models: 
        + [ ] AutoRegression (AR),  Moving Average (MA) ,  AR+MA model (ARMA) ARIMA model

**Preprocessing**
- [ ] Normalization, Standardization
- [ ] Data transformation: 
    + [ ] PCA, ICA ,LDA 
    + [ ] TF-IDF
    + [ ] One Hot Encoding
    + [ ] Bag of Word

**Machine Learning models**
- [ ] Basic model: **Linear model and Non-linear model** 
    + [x] Linear Regression
    + [x] Logistic Regression
    + [ ] Decision Tree
    + [ ] Random Forest
    + [ ] K-mean Clustering
    + [ ] k-NN
    
- [ ] Ensemble Learning
    + [ ] Bagging
    + [ ] Boosting Machine (Basic, Adaptive Boosting)
    + [ ] Stacking(Optional)

**Model Evaluation and Optimization**
- [ ] Model Error: variance, bias
- [ ] Model Loss function:
    + [ ] Least Mean Square Loss (Root mean square loss, sum square loss)
    + [ ] Cross Entropy
    + [ ] K-L divergence loss (Advanced)
    + [ ] Hinge Loss (Advanced)

- [ ] Model optimization/training methods:
    + [ ] Stochastic Gradient descent
    + [ ] Adam
    + [ ] Adaptive Optimizer
    + [ ] Momentum Second order optimizer

- [ ] Model performance:
    + [ ] ROC, AUC
    + [ ] precision
    + [ ] recall/sensitivity
    + [ ] Specificity
    + [ ] F-1 score

- [ ] Estimate performance
    + [ ] hold-out
    + [ ] k-fold Cross-validation
    + [ ] Stratified k-fold cross-validation
    + [ ] leave-one-out cross-validation

- [ ] Methods for avoiding Overfitting:
    + [ ] Regularization: L1 (lasso regression), L2 (Ridge Regression)
    + [ ] Cross-validation
    + [ ] Ensemble learning: bagging (randomforest), boosting, stacking
    + [ ] Prunning: pre-pruning, post-prunning
    + [ ] Dropout (Deep learning)

- [ ] Methods for accelerating learning
    + [ ] BatchNorm
    + [ ] Normalization, Standardization

**Feature Selection**
- [ ] L1 Regularization
- [ ] Feature importance 
- [ ] Recursive feature removing / adding
- [ ] K-mean clustering for new feature


**Statistic**
- [ ] Descriptive Statistic for data analysis
    + [ ] univariate: analyze mean, spread, distribution of single variable
    + [ ] bivariate: correlation between two variable
    + [ ] multi-variate

- [ ] Inference based statistic
    + [ ] how to choose Mean or Medium
    + [ ] Null hypothesis and Hypothesis testing
    + [ ] p-value, t-test, z-test, ANOVA test(optional)
    + [ ] Chi Square, Correlation
- [ ] A/B Testing
    + [ ] Define Metrices
    + [ ] Randomization / Random experience
    + [ ] AA testing
- [ ] Hypothesis Testing
    + [ ] T-statistic
    + [ ] F-statistic
    + [ ] Z-Statistic / z-score
    + [ ] Chi-distribution / measure independece


**Deep Learning**
- [ ] Different Activation Functions and gradient vanishing
- [ ] Weight Decay (add regularization term)
- [ ] Backpropagation
- [ ] Weight Initialization
    + [ ] zero- initialization (symmetric property)
    + [ ] 1/sqrt(n) initialization
- [ ] CNN
- [ ] Batch Norm
- [ ] Pooling
- [ ] Dropout (Average, max)
- [ ] Methods to avoid gradient vanishing (use sum term F(x) + x)
- [ ] Methods to avoid gradient explosion (gradient Click)
- [ ] RNN and BPTT (backpropagation through time)
- [ ] LSTM
- [ ] Gated Recurrent Unit (GRU)
- [ ] Attention
    - [ ] Self-Attention
    - [ ] multi-head attention
- [ ] Embedding
    + [ ] Word2Vector, Optimization in Word2Vector (Negative Sampling, SGD with Sampling)
    + [ ] Transformer: BERT, GPT, embedding

- [ ] Inference Strategies (Accelerate inference and save memory)
    + [ ] Post-prunning after model training (delete neuron/activation outputs and synapses/weights)
    + [ ] Quantization (round the weight with fewer bits to decrease model size)
    + [ ] Kernel method (like SVM kernel)

**Recommendation System Models**
- [ ] Collaborate Filtering
    + [ ] User-based 
    + [ ] Item-based
    + [ ] Content-based
    + [ ] Cold Startup Strategy
- [ ] Matrix Factorization / Alternative Least Square Algorithm (ALS)
- [ ] Wide and Deep Deep learning Model (Deep CTR)

**Data Engineer**
- [ ] Distributed Computing and Storage
    + [ ] Programming model (Master-slave, product-consumer, work queue)
    + [ ] Hadoop Distributed File System (HDFS)
    + [ ] Map, Reduce: Distributed Computing
    + [ ] Difference between PySpark and Hadoop in distributed computing
- [ ] Data Streaming
    + [ ] KafKa: subscript and collect data (product-consumer programming model)
    + [ ] Flink Streaming / pyspark  structure Streaming framework



**Business Analysis**
- [ ] Customer Journey
- [ ] Profit-Cost model framework to analysis the reason of performance


**Business Question**
- [ ] Reason for applying this job and company
- [ ] Communication
- [ ] Challenging project/ thing
- [ ] My Drawbacks and Advantages and Improvement on Drawbacks
- [ ] Case Study/Real problem for applying data analysis

#### Overall
- [ ] Programming Language: Python, SQL, C++/Java
- [ ] Data Structure
- [ ] Searching (DFS, BFS, DP)
- [ ] Statistic and Possibility
- [ ] Platforms
    + [ ] Data streaming (Data collection tools): KafKa, PySpark/Flink Streaming
    + [ ] Distributed Computing (Data processing platforms): PySpark, hadoop
    
- [ ] Data Pipeline
    + [ ] Problem Definition and Clarification
    + [ ] Collect Data
    + [ ] **Data Cleaning and Preprocessing /Extract, Transform, Load (ETL)** in data warehouse (数据仓库)
    + [ ] Exploratory Data Analysis **(EDA)** for features exploration
    + [ ] Feature Selection (some models have feature selection function)
    + [ ] Machine Learning Modeling 
    + [ ] Model Evaluation and Selection
    + [ ] Model Improvement
    + [ ] Model Inference and Deploy Model 

- [ ] Business Question adn Communication Skills and Project/Experience Descriptions
    
