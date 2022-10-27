from numpy import *
from scipy.stats import norm
from scipy.optimize import root, fsolve, newton
import logging

"""
Reference: https://gist.github.com/neerav1985/8291563
The code is referred from the GitHub, to calculate the implied volatility given the parameters
Modification is done for fitting into application use.
"""

"""
Stage: Stage 2 file

Document type: Preprocessing only.

Need to run? No.

Main purpose: to calculate the implied volatility and the Greeks in options.

Dependency:
    use -> None
    be used -> OptionExtraction.py
    
Methods:
        BlackScholesCall: to calculate the reasonable call price based on the BS Model
        BlackScholesPut: to calculate the reasonable put price based on the BS Model
        Delta: get delta value of the Greeks
        Gamma：get gamma value of the Greeks
        Vega: get vega value of the Greeks
        Theta：get theta value of the Greeks
        GetImpVol: get Implied volatility based on the optimization method
"""

def BlackScholesCall(S, K, T, sigma, r = 0., q = 0.):
        """
        to calculate the reasonable call price based on the BS Model
        :param S: spot price
        :param K: strike price
        :param T: time to maturity
        :param sigma: implied volatility
        :param r: risk-free rate
        :return: the reasonable price for the call option
        """
        d1 = (1/(sigma*sqrt(T)))*(log(S/K) + (r + sigma**2)*T)
        d2 = d1 - sigma*sqrt(T)
        return S*exp(-q*T)*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)

def BlackScholesPut(S, K, T, sigma, r = 0., q = 0.):
        """
        to calculate the reasonable put price based on the BS Model
        :param S: spot price
        :param K: strike price
        :param T: time to maturity
        :param sigma: implied volatility
        :param r: risk-free rate
        :return: the reasonable price for the call option
        """
        d1 = (1/(sigma*sqrt(T)))*(log(S/K) + (r + sigma**2)*T)
        d2 = d1 - sigma*sqrt(T)
        return  (-S*exp(-q*T)*norm.cdf(-d1)) + (K*exp(-r*T)*norm.cdf(-d2))

def Delta(sigma, S, K, T,r, q, pricer):
        """
        get delta value of the Greeks
        :param S: spot price
        :param K: strike price
        :param T: time to maturity
        :param sigma: implied volatility
        :param r: risk-free rate
        :param pricer: is BlackScholesCall or BlackScholesPut
        :return: the reasonable price for the call option
        """
        h = 1.e-3
        return ((pricer(S + h, K, T, sigma, r, q) - pricer(S - h, K, T, sigma , r, q))/2) / h

def Gamma(sigma, S, K, T,r, q, pricer):
        """
        get gamma value of the Greeks
        :param S: spot price
        :param K: strike price
        :param T: time to maturity
        :param sigma: implied volatility
        :param r: risk-free rate
        :param pricer: is BlackScholesCall or BlackScholesPut
        :return: the reasonable price for the call option
        """
        h = 1.e-2
        return (pricer(S + h, K, T, sigma , r, q) - 2* pricer(S , K, T, sigma, r, q) + pricer(S -h, K, T, sigma, r, q))/h**2

def Vega(sigma, S, K, T,r, q, pricer):
        """
        get Vega value of the Greeks
        :param S: spot price
        :param K: strike price
        :param T: time to maturity
        :param sigma: implied volatility
        :param r: risk-free rate
        :param pricer: is BlackScholesCall or BlackScholesPut
        :return: the reasonable price for the call option
        """
        h = 0.01
        return (pricer(S, K, T, sigma + h, r, q) - pricer(S, K, T, sigma - h, r, q))/2

def Theta(sigma, S, K, T,r, q, pricer):
        """
        get theta value of the Greeks
        :param S: spot price
        :param K: strike price
        :param T: time to maturity
        :param sigma: implied volatility
        :param r: risk-free rate
        :param pricer: is BlackScholesCall or BlackScholesPut
        :return: the reasonable price for the call option
        """
        h = 1 / 365
        return (pricer(S, K, T-h, sigma, r, q) - pricer(S, K, T, sigma, r, q))

#Get the error of the option price based on the BS model with implied volatility in assumption.
option_err = lambda sigma, option_price, S, K, T, r, q, pricer: abs(pricer(S, K, T, sigma, r, q) - option_price)

def GetImpVol(sigma0, call_price, S, K, T, r, q, pricer, method ):
        """
        To get the implied volatility
        :param sigma0: the target output, implied volatility
        :param call_price:
        :param S: spot price
        :param K: strike price
        :param T: time to maturity
        :param r: risk-free rate
        :param pricer: is BlackScholesCall or BlackScholesPut
        :param method: The optimized method
        :return: The implied volatility after optimization
        """
        dict = {"root":root, "fsolve": fsolve}
        return dict[method]( option_err , sigma0, args = (call_price, S, K, T, r, q, pricer))


