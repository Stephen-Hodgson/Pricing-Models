# Pricing-Models

In order to develop my understanding of pricing models and gain practical experience in modelling, I use various mathematical and computational methods to value different financial instruments.

Random_Processes.py is used to simulate random processes that can be used to model stock prices, interest rate models and more. This contains the class BrownianMotion which I can use to simulate random walks and CompoundBM which can be used to create correlated Brownian motions. JumpProcess has not yet been implemented

Rate_Models.py contains interest rate models. These include constant rates models and time-varying non-random rate models. Affine yield models will be introduced soon. From the interest rate models, we can calculate zero-coupon bond values

Volatility_Models.py contains volatility models. These currently include constant rates models and time-varying non-random rate models.

Underlying.py contains prices of assets. This currently only models Stock Prices which take a volatility model, rate model and brownian motion to generate a stock price. Dividends can be paid out at a fixed rate or in lump payments at specific times. The exponential martingale exp(vol*W-vol**2/2 * t) is checked using a Monte-Carlo method with importance sampling.

Vanillas.py and Exotics.py are used to model option pricing. Currently this is only done via Black-Scholes-Merton equations for constant or non-random rate and volatility models but pricing via Monte-Carlo simulation or integration methods are planned

Upcoming plans:

One-factor Vasicek rate models
Monte Carlo simulation for option pricing with random interest rate
American style-options

Note: All code here is written by me for my own education. The reader may use/copy this code as they wish.

References used:
Steven E. Shreve Stochastic Calculus for Finance Vol. 1 and 2