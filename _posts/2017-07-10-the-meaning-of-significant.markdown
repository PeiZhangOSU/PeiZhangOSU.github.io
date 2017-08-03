---
layout: post
title:  "The meaning of 'significant'"
date:   2017-07-10 10:38:00 -0400
tags:
- Statistics
- Misc
---

I am sure all researchers have encountered and/or performed hypothesis testing at one point or another. Even if you are not a researcher by trade, it is very likely you have heard the term "p-value" or "the result is significant" from the news.

But what does "significant" mean? <!--more--> According to [wikipedia](https://en.wikipedia.org/wiki/Statistical_significance), statistical significance means "the significance level defined for a study, α, is the probability of the study rejecting the null hypothesis, given that it were true; and the p-value of a result, p, is the probability of obtaining a result at least as extreme, given that the null hypothesis were true. The result is statistically significant, by the standards of the study, when p < α." Not exactly simple and straightforward, which is probably one of the reasons for the many confusion and misunderstandings about what is "significant".

There are numerous articles explaining the meaning of statistical significance and correcting its common misinterpretations <sup>[1](#myfootnote1)</sup>. Recently, I read a good explanation from an etymology perspective in the book "The Lady Tasting Tea - How Statistics Revolutionized Science in the Twentieth Century" by David Salsburg:

>Somewhere early in the development of this general idea, the word *significant* came to be used to indicate that the probability was low enough for rejection. Data became significant if they could be used to reject a proposed distribution. The word was used in its late-nineteenth-century English meaning, which is simply that the computation signified or showed something. As the English language entered the twentieth century, the word significant began to take on other meanings, until it developed its current meaning, implying something very important. Statistical analysis still uses the word significant to indicate a very low probability computed under the hypothesis being tested. In that context, the word has an exact mathematical meaning. Unfortunately, those who use statistical analysis often treat a significant test statistic as implying something much closer to the modern meaning of the word.

It is helpful to remember that in the context of statistics, "significant" originally meant "able to signify", instead of "important". It does not indicate anything about the plausibility of the hypothesis nor the importance of the results.

##### <a name="myfootnote1">1</a>: In particular, I would like to mention this paper titled "Statistical tests, P values, confidence intervals, and power: a guide to misinterpretations" (Greenland, S., Senn, S.J., Rothman, K.J. et al. Eur J Epidemiol (2016) 31: 337. doi:10.1007/s10654-016-0149-3). It lists 25 common misinterpretations of hypothesis testing, and explains why they are wrong.
