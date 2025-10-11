# Uniform Distribution
# use when : all outcomes are equally likely within a range
from scipy.stats import uniform
import numpy as np
import matplotlib.pyplot as plt

a,b=0,10 # lower and upper bounds
x=np.linspace(a-1, b+1, 1000) # 1000 points from a-1 to b+1

# PDF: probability density function (height of cirve)
pdf=uniform.pdf(x, loc=a, scale=b-a) # scale = b-a
# CDF: cumulative distribution function (area under curve)
cdf=uniform.cdf(x, loc=a, scale=b-a)

plt.figure(figsize=(8,4)) 
plt.subplot(1,2,1)
plt.plot(x,pdf,color='Blue')
plt.title('Uniform Distribution PDF')
plt.xlabel('x')
plt.ylabel('Density f(x)')

plt.subplot(1,2,2)
plt.plot(x,cdf,color='blue')
plt.title('Uniform Distribution CDF')
plt.xlabel('x')
plt.ylabel('F(x) = P(X ≤ x)')
plt.tight_layout()
plt.show()

# Normal Distribution
# use when : data clusters around a mean (bell curve)

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 0, 1  # standard normal
x = np.linspace(-4, 4, 500)

pdf = norm.pdf(x, mu, sigma)  # bell curve
cdf = norm.cdf(x, mu, sigma)  # cumulative curve

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(x, pdf, color='purple')
plt.title("PDF of N(0,1)")
plt.xlabel("x"); plt.ylabel("Density f(x)")

plt.subplot(1, 2, 2)
plt.plot(x, cdf, color='orange')
plt.title("CDF of N(0,1)")
plt.xlabel("x"); plt.ylabel("F(x)")
plt.tight_layout()
plt.show()
