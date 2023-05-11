---
layout: post
comments: true
title: Double Machine Learning for Pricing Elasticity Estimation
author: Shaodong Wang
---

## What is pricing elasticity
Pricing elasticity describes the sensitivity of demand to changes in price for a particular product. When demand is more elastic, an increase in price will result in a greater reduction in demand. This concept allows businesses to estimate how many more units of a product they could sell if they were to lower the price by a certain percentage.

In short, economists summarized the elasticity to be a simple equation: 

$$\theta = \frac{\partial Q / Q}{\partial P / P}$$

or equivalently,

$$log Q \sim \theta log P$$

where $$P$$ is price, $$Q$$ is demand, and $$\theta$$ is the elasticity. This equations tells us that given a percent-change of price ($$P$$), the percent-change of demanded quantity ($$Q$$) is a constant. This constant is the elasticity, $$\theta$$. 

The basic idea is that a \\$1 increase in price will have a larger impact on demand for a product that costs \\$5 compared to one that costs \\$100. Consumers tend to care about relative changes rather than absolute changes. This definition is convenient because it allows the parameter $$\theta$$ to remain constant as the price changes. With a reliable estimate of $$\theta$$, a retailer can make counterfactual predictions about their prices, such as "if I were to increase the price of my product by 5%, I could sell 5θ% more units" (usually $$\theta$$ is negative).

A good elasticity estimation can be very important to a retailer in many scenarios:

- Pricing Strategy: A retailer with a good understanding of price elasticity can optimize their pricing strategy to maximize revenue. For example, if a product has low elasticity, the retailer can increase the price without worrying too much about a decrease in demand. On the other hand, for a highly elastic product, lowering the price might lead to a significant increase in demand, leading to higher revenue overall.

- Promotional Offers: Retailers can use price elasticity to determine the best promotional offers for their products. For instance, if a product has high elasticity, a small discount might result in a big increase in demand. However, for a product with low elasticity, a larger discount might be necessary to encourage consumers to buy.

- Inventory Management: Retailers can use price elasticity to help manage their inventory. For example, if they know that a product is highly elastic and likely to sell out quickly if the price is lowered, they may decide to keep a smaller inventory and replenish it more frequently.

- Market Analysis: By examining price elasticity across different products and markets, a retailer can gain insights into consumer behavior and preferences. This information can be used to inform pricing, marketing, and product development decisions.

## Why do we need double ML for pricing elasticity estimation
Although a good elasticity estimation can be very important to a retailer, it is challenging to estimate the pricing elasticity coefficient in the real world.  

We probability want to estimate elasticity from historical selling data. But historical selling data can lead to a biased elasticity if we do not remove the confounding factors, for example, holiday. In many industries, low prices are associated with low sales. For example, in the hotel industry, prices are low outside the tourist season, and prices are high when demand is highest and hotels are full. Given that data, a naive estimation might suggest that increasing the price would lead to more sales.

To remove the confounding effects, the most straightforward way is to set up a randomized experiments. For instance, a retailer could randomly adjust product prices up and down or even randomize prices across different customers. Then with the collected price and demand data, we are easily able to estimate the pricing elasticity. However, this type of experimentation is not good or realistic.
 - The experiment is expensive as it requires selling products at suboptimal prices and can negatively impact the customer experience. 
 - Usually, a short experiment might not be generalizable to other seasons or holidays.

Given the non-randomized observations, how do we remove the confounding effects and estimate the true effect of price on sales? The answer is Double Machine Learning (DML). 


## How does double ML work
Pricing elasticity estimation is not just fitting a machine learning model using the available data. As we said in the last section, a naive estimation may give ridiculous suggestions due to the biased data. 

Double machine learning is a method that combines machine learning algorithms to estimate treatment effects in causal inference. It aims to answer the questions of 'what if'. For example, what sales would be if I set the discount to 30%? 

Generally, DML include two steps. 1) In the first step, we train two separate models to predict the treatment (price) and outcome (sales) using confounding variables, respectively. 2) In the second step, we estimate the pricing elasticity on the residuals of price and sales from the trained models in the first step. 

Specifically, we want to estimate causal effect, $$\theta$$ using the following equations:

$$Y = \theta T + G(W) + \epsilon$$

where $$W$$ is the confounding variables that can impact both $$Y$$ and $$T$$.

This equation holds after transformation.

$$Y-E[Y|W] = \theta (T - E[T|W]) + \epsilon$$

It tells us that the treatment effect can be derived from regression on residuals ([Frisch-Waugh-Lovell theorem](https://en.wikipedia.org/wiki/Frisch%E2%80%93Waugh%E2%80%93Lovell_theorem))! 

Based on this finding, DML estimates treatment effect through the [following procedure](https://matheusfacure.github.io/python-causality-handbook/22-Debiased-Orthogonal-Machine-Learning.html):
    1. Estimate the outcome $$Y$$ with confounding variables $$W$$ using a flexible ML regression model $$M_y$$.
    2. Estimate the treatment $$T$$ with features confounding variables $$W$$ using a flexible ML regression model $$M_t$$.
    3. Obtain the residuals $$\tilde{Y}=Y-M_y(W)$$ and $$\tilde{T}=T-M_t(W)$$.
    4. Regress the residuals of the outcome on the residuals of the treatment $$\tilde{Y}=\theta \tilde{T}+\epsilon$$





That's why double ML can estimate the causal effect by building models on residuals.
To understand this equation, we can do one more transformation: 

This formula says that we can predict T from X. After we do that, we’ll be left with a version of T, 
, which is uncorrelated with all the variables included previously. This will break down arguments such as “people that have more years of education (T) have it because they have higher X. It is not the case that education leads to higher wages. It is just the case that it is correlated with X, which is what drives wages”. Well, if we include X in our model, then 
 becomes the return of an additional year of education while keeping X fixed. 

## Elasticity for a group of products

## Beyond pricing elasticity
Given a good estimation of pricing elasticity, how do we make use of other variables?







## References
[1] https://towardsdatascience.com/causal-inference-example-elasticity-de4a3e2e621b

https://doi.org/10.1016/j.eswa.2013.07.059 / 
https://isiarticles.com/bundles/Article/pre/pdf/40549.pdf

https://arxiv.org/pdf/2205.01875.pdf

https://www.actable.ai/use-cases/optimizing-sales-causal-inference-could-be-your-secret-sauce

(good) https://github.com/larsroemheld/causalinf_ex_elasticity/blob/main/elasticity_dml.ipynb 
