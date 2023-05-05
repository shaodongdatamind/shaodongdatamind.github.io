---
layout: post
comments: true
title: Causal Inference
author: Shaodong Wang
---


## causal inference rule of thumb for regression
As a rule of thumb, always include confounders and variables that are good predictors of 
 in your model. Always exclude variables that are good predictors of only 
, mediators between the treatment and outcome or common effect of the treatment and outcome.
