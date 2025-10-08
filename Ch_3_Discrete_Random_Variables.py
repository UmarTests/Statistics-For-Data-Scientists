# Bernoulli Distribution
#use when : yes/ no , spam/ not spam, success/ failure
from scipy.stats import bernoulli
import matplotlib.pyplot as plt

p=0.7
x=[0,1]
pmf=bernoulli.pmf(x,p)

plt.bar(x,pmf)
plt.title('Bernoulli Distribution (p=0.7)')
plt.xlabel('Outcomes')
plt.ylabel('Probability')
plt.xticks(x)
plt.show()

# Binomial Distribution
#use when : number of successes in n independent Bernoulli trials
from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt

n,p = 10, 0.5
x=np.arange(0, n+1)
pmf=binom.pmf(x,n,p)

plt.bar(x,pmf)
plt.title('Binomial Distribution (n=10, p=0.5)')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.show()

# Geometric Distribution
#use when : number of trials until first success
from scipy.stats import geom
import numpy as np
import matplotlib.pyplot as plt

p=0.4
x=np.arange(1,15)
pmf=geom.pmf(x,p)

plt.bar(x,pmf)
plt.title('Geometric Distribution (p=0.4)')
plt.xlabel('Number of Trials until First Success')
plt.ylabel('Probability')
plt.show()

# Poisson Distribution
#use when : number of events in a fixed interval of time/ space
from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt

# lambda = average rate of occurrence
lmbda=5
# x= possible number of events
x=np.arange(0,25)
# pmf() gives the prob of each count
pmf=poisson.pmf(x,lmbda)

plt.bar(x,pmf, color='skyblue', edgecolor='black')
plt.title('Poisson Distribution (λ=5)')
plt.xlabel('Number of Events')
plt.ylabel('Probability P(X=k)')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()