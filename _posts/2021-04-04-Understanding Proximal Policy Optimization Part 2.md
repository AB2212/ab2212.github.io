---
layout:     post
title:      "Understanding Proximal Policy Optimization Part 2 "
subtitle:   "Diving deeper into Importance Sampling, Trust Region Policy Optimization and Clipped Surrogate Objective function"
date:       2021-04-04 12:00:00
author:     "Abhijeet Biswas"
header-img: "img/night_sky_ppo1.jpg"
---

Welcome to the second part of this three-part blog series where we deep dive into the theory and implementation details behind Proximal Policy Optimization (PPO) in PyTorch. In the first part of the series, we understood what Policy Gradient methods are; in the second part we will look into recent developments in Policy Gradient methods like Trust Regions and Clipped Surrogate Objective to understand how PPO works; in the third part we will go through the detailed implementation of PPO in Pytorch where we will teach an agent to land a rocket in gym's [Lunar Lander](https://gym.openai.com/envs/LunarLander-v2/) environment and also help a [Bipedal Walker](https://gym.openai.com/envs/BipedalWalker-v2/) learn to walk. Let's get started!
<figure>
  <video controls width="100%" src="{{ site.baseurl }}/img/ppo_video.mp4" autoplay loop/>
</figure>

**Table of Content**
1. TOC
{:toc}

<p> </p>
In case you have missed the first part, click [here](https://www.digdeepml.com/2020/05/24/Understanding-Proximal-Policy-Optimization-Part-1/). So far we have looked into what policy gradient methods are and how we can use various baselines to reduce variance which gives rise to different methods. In this part, we will look into what makes PPO sample efficient and helps it to deliver reliable performance without much hyperparameter tuning. We will start with understanding the limitations of vanilla policy gradient methods and then go through the various improvements that led the development of PPO.
### Limitations of Vanilla Policy Gradient Methods
#### Step size

In policy gradient methods, the input data is non-stationary due the changing policy which in turn changes observation and reward distribution. This makes it hard to take different step sizes for getting good policy while optimizing the parameters. Initially, if we have a bad policy then we might have to take  big step sizes but later on we need to reduce our step size as we don't want to deviate from the good policy. There is no consistent good step size that works throughout the learning process which makes it hard.

Step too far &#8594; Bad Policy &#8594; Bad Input Data &#8594; Difficult or no recovery

#### Sample Efficiency
We use data sample only once to calculate a single gradient step for model parameter update. We don't use all the available information out of it. For example, we may have some small gradients in our parameter update because of poor scaling of features, then we would make very little progress on the parameters with tiny gradients and hence slow learning

Machine learning is all about reducing learning to numerical optimization. If we can change the loss function that we are optimizing we might be able to find solutions to these problems. Let's start with rewriting the objective function using importance sampling to tackle these issues. But what is Importance Sampling?

