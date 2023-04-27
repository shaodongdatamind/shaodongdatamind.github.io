---
layout: post
comments: true
title: Double Machine Learning for Pricing Elasticity Estimation
author: Shaodong Wang
---

## What is pricing elasticity

## Why do we need double ML for pricing elasticity estimation
The following example is from [1]. 
The ability to learn causally valid elasticity from observational data is therefore key; observational data in this example is simply the retailer’s history of prices and units sold over time. But estimating causal effects from observational data is difficult because of confounding. To see what this means, consider (1) product quality and (2) seasons as two important examples of many potential confounders:

MacBooks are more expensive than, say, Chromebooks. Assuming a retailer sells more MacBooks than Chromebooks (and nothing else), the observational data indicates that high prices correlate with high sales. But it would be foolish to (counterfactually) expect that raising a Chromebook’s price to Apple- levels would allow selling more Chromebooks.
Demand for many products is seasonal, for example due to holidays (Christmas) and weather changes (summer). Typically, prices are high during high season (and yet a lot of products are sold), and lower during off-season (when fewer products are sold); yet despite this correlation it would be foolish to expect higher sales from raising off-season prices to high season levels.
As the saying goes, the retailer must be careful not to confuse correlation and causation. The following causal graph represents a simple confounder relationship: failing to control for product quality (and season, and others, not displayed) will significantly bias estimates of θ. Such biases will lead the retailer to wrong conclusions about optimal prices, directly hurting their business.

## How does double ML work
We want to estimate causal effect, $$\theta$$ using the following equations:

$$Y = \theta T + G(X) + \epsilon$$

$$T = X\beta + \eta$$

This equation holds after transformation.

$$Y-E[Y|X] = \theta (T - E[T|X]) + \epsilon$$
That's why double ML can estimate the causal effect by building models on residuals.



## Beyond pricing elasticity
Given a good estimation of pricing elasticity, how do we make use of other variables?







## References
[1] https://towardsdatascience.com/causal-inference-example-elasticity-de4a3e2e621b

https://doi.org/10.1016/j.eswa.2013.07.059 / 
https://isiarticles.com/bundles/Article/pre/pdf/40549.pdf

https://arxiv.org/pdf/2205.01875.pdf

https://www.actable.ai/use-cases/optimizing-sales-causal-inference-could-be-your-secret-sauce

(good) https://github.com/larsroemheld/causalinf_ex_elasticity/blob/main/elasticity_dml.ipynb 
