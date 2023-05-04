---
layout: post
comments: true
title: Double Machine Learning for Pricing Elasticity Estimation
author: Shaodong Wang
---

## What is pricing elasticity
Pricing elasticity describes the sensitivity of demand to changes in price for a particular product. When demand is more elastic, an increase in price will result in a greater reduction in demand. This concept allows businesses to estimate how many more units of a product they could sell if they were to lower the price by a certain percentage.

In short, economists summarized the elasticity to be a simple equation: 

$$\theta = \frac{\partial q / q}{\partial p / p}$$, 

where $$p$$ is price, $$q$$ is demand, and $$\theta$$ is the elasticity. This equations tells us that given a percent-change of price ($$p$$), the percent-change of demanded quantity ($$q$$) is a constant. This constant is the elasticity, $$\theta$$. 

The basic idea is that a $1 increase in price will have a larger impact on demand for a product that costs $5 compared to one that costs $100. Consumers tend to care about relative changes rather than absolute changes. This definition is convenient because it allows the parameter $$\theta$$ to remain constant as the price changes. With a reliable estimate of $$\theta$$, a retailer can make counterfactual predictions about their prices, such as "if I were to increase the price of my product by 5%, I could sell 5θ% more units" (usually θ is negative).


## Why do we need double ML for pricing elasticity estimation

Causal inference answers the questions of 'what if'

https://matheusfacure.github.io/python-causality-handbook/01-Introduction-To-Causality.html

ML is notoriously bad at this inverse causality type of problem. They require us to answer “what if” questions, which economists call counterfactuals. What would happen if I used another price instead of this price I’m currently asking for my merchandise? What would happen if I do a low sugar one instead of this low-fat diet I’m in? If you work in a bank, giving credit, you will have to figure out how changing the customer line changes your revenue. Or, if you work in the local government, you might be asked to figure out how to make the schooling system better. Should you give tablets to every kid because the era of digital knowledge tells you to? Or should you build an old-fashioned library?

The following example is from [1]. 
The ability to learn causally valid elasticity from observational data is therefore key; observational data in this example is simply the retailer’s history of prices and units sold over time. But estimating causal effects from observational data is difficult because of confounding. To see what this means, consider (1) product quality and (2) seasons as two important examples of many potential confounders:

MacBooks are more expensive than, say, Chromebooks. Assuming a retailer sells more MacBooks than Chromebooks (and nothing else), the observational data indicates that high prices correlate with high sales. But it would be foolish to (counterfactually) expect that raising a Chromebook’s price to Apple- levels would allow selling more Chromebooks.
Demand for many products is seasonal, for example due to holidays (Christmas) and weather changes (summer). Typically, prices are high during high season (and yet a lot of products are sold), and lower during off-season (when fewer products are sold); yet despite this correlation it would be foolish to expect higher sales from raising off-season prices to high season levels.
As the saying goes, the retailer must be careful not to confuse correlation and causation. The following causal graph represents a simple confounder relationship: failing to control for product quality (and season, and others, not displayed) will significantly bias estimates of θ. Such biases will lead the retailer to wrong conclusions about optimal prices, directly hurting their business.

Another good example:

in many industries, low prices are associated with low sales. For example, in the hotel industry, prices are low outside the tourist season, and prices are high when demand is highest and hotels are full. Given that data, a naive prediction might suggest that increasing the price would lead to more rooms sold.

## How does double ML work
We want to estimate causal effect, $$\theta$$ using the following equations:

$$Y = \theta T + G(X) + \epsilon$$

$$T = X\beta + \eta$$

This equation holds after transformation.

$$Y-E[Y|X] = \theta (T - E[T|X]) + \epsilon$$
That's why double ML can estimate the causal effect by building models on residuals.
To understand this equation, we can do one more transformation: 

This formula says that we can predict T from X. After we do that, we’ll be left with a version of T, 
, which is uncorrelated with all the variables included previously. This will break down arguments such as “people that have more years of education (T) have it because they have higher X. It is not the case that education leads to higher wages. It is just the case that it is correlated with X, which is what drives wages”. Well, if we include X in our model, then 
 becomes the return of an additional year of education while keeping X fixed. 

## Beyond pricing elasticity
Given a good estimation of pricing elasticity, how do we make use of other variables?







## References
[1] https://towardsdatascience.com/causal-inference-example-elasticity-de4a3e2e621b

https://doi.org/10.1016/j.eswa.2013.07.059 / 
https://isiarticles.com/bundles/Article/pre/pdf/40549.pdf

https://arxiv.org/pdf/2205.01875.pdf

https://www.actable.ai/use-cases/optimizing-sales-causal-inference-could-be-your-secret-sauce

(good) https://github.com/larsroemheld/causalinf_ex_elasticity/blob/main/elasticity_dml.ipynb 
