---
layout: post
comments: true
title: Causal Inference
author: Shaodong Wang
---

https://matheusfacure.github.io/python-causality-handbook/07-Beyond-Confounders.html

## causal inference rule of thumb for regression
As a rule of thumb, always include confounders and variables that are good predictors of Y
 in your model. Always exclude variables that are good predictors of only T
, mediators between the treatment and outcome or common effect of the treatment and outcome.

Examples of selection bias that sound reasonable:
- 1. Adding a dummy for paying the entire debt when trying to estimate the effect of a collections strategy on payments.
- 2. Controlling for white vs blue collar jobs when trying to estimate the effect of schooling on earnings
- 3. Controlling for conversion when estimating the impact of interest rates on loan duration
- 4. Controlling for marital happiness when estimating the impact of children on extramarital affairs
- 5. Breaking up payments modeling E[Payments] into one binary model that predict if payment will happen and another model that predict how much payment will happen given that some will: E[Payments|Payments>0]*P(Payments>0)
