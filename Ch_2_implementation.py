# Law of Large Numbers (LLN) Simulation
'''We wanna see that if we flip a fair coin a lot, 
the average heads probability converges to 0.5.'''

import numpy as np
import matplotlib.pyplot as plt

# simulate coin flips
N = 10000  # number of flips
flips = np.random.randint(0, 2, N)  # 0 or 1
cum_avg = np.cumsum(flips) / np.arange(1, N+1)

# Plot Convergence
plt.figure(figsize=(10,5))
plt.plot(cum_avg, label="Sample Mean")
plt.axhline(0.5, color='red', linestyle='--', label="True Probability")
plt.xlabel("Number of Flips")
plt.ylabel("Proportion of Heads")
plt.title("Law of Large Numbers in Action")
plt.legend()
plt.show()

'''Observation (LLN experiment):
When the number of flips is small, the sample mean (proportion of heads) bounces around 
a lot due to randomness. But as the number of flips grows large, the sample mean stabilizes 
and converges to the true probability (0.5). This illustrates 
the Law of Large Numbers: sample averages approximate expected values as the sample size increases.'''



# ----The Central Limit Theorem (CLT)---
'''if you take many samples, compute their averages, and plot them, the distribution of those averages 
starts looking like a bell curve (normal distribution) — no matter the original distribution.    '''
import numpy as np
import matplotlib.pyplot as plt

# Simulation
'''Flip a fair coin (0 or 1) many times.
Group flips into samples of size n.
Compute the mean of each sample.
Repeat many times to see the shape.'''

# Parameters
n = 30        # sample size
num_samples = 10000  # how many times we repeat

# Simulate
samples = np.random.randint(0, 2, (num_samples, n))
sample_means = samples.mean(axis=1)
# Plotting
plt.figure(figsize=(10,5))
plt.hist(sample_means, bins=20, density=True, alpha=0.7, color="blue")
plt.title(f"Central Limit Theorem with Coin Flips (n={n})")
plt.xlabel("Sample Mean")
plt.ylabel("Density")
plt.show()

'''Observation (CLT experiment):
Even though individual coin flips are just 0 or 1 (a discrete, non-bell-shaped distribution),
when we take many samples of size n and average them, the histogram of those sample means 
forms a bell-shaped curve centered at the true mean (0.5). This demonstrates 
the Central Limit Theorem: regardless of the original distribution, the distribution of 
sample means tends toward a normal distribution as n increases.'''




# ----Independence Check via Simulation---
'''We can simulate two independent random variables and check their independence
by looking at their joint distribution.'''

# We’ll use dice  (since they’re intuitive)
import numpy as np

N = 100000  # number of trials
die1 = np.random.randint(1, 7, N)
die2 = np.random.randint(1, 7, N)

# print(die1[:10], die2[:10])  # first 10 rolls

A = (die1 % 2 == 0)             # first die is even
B = ((die1 + die2) > 7)         # sum > 7

P_A = A.mean()
P_B = B.mean()
P_A_and_B = (A & B).mean()

print("P(A) =", P_A)
print("P(B) =", P_B)
print("P(A and B) =", P_A_and_B)
print("P(A)*P(B) =", P_A * P_B)

'''Interpretation (Independence check):

𝑃(𝐴)≈0.50
P(A)≈0.50 → makes sense, half the time the first die is even.
𝑃(𝐵)≈0.42
P(B)≈0.42 → about 42% of rolls have sum > 7.
If A and B were independent, we’d expect

𝑃(𝐴 ∩ 𝐵)≈𝑃(𝐴) ⋅𝑃(𝐵)≈0.21
P(A∩B)≈P(A)⋅P(B)≈0.21

But the simulation gave 
𝑃(𝐴 ∩ 𝐵)≈0.25
P(A∩B)≈0.25, which is clearly bigger than 0.21.
Conclusion: Events A and B are not independent. Knowing that the first die 
is even actually increases the chance the sum exceeds 7.'''


# ----Birthday Problem (Probability Paradox)---
'''We want the probability that in a group of n people, 
at least two share the same birthday.'''

import numpy as np

def birthday_sim(n_people, n_trials=10000):
    count = 0
    for _ in range(n_trials):
        birthdays = np.random.randint(1, 366, n_people)  # assume 365 days
        if len(set(birthdays)) < n_people:  # duplicate found
            count += 1
    return count / n_trials

for n in [5, 10, 20, 23, 30, 50]:
    print(f"n={n}, P(shared birthday) ≈ {birthday_sim(n):.3f}")

'''Interpretation (Birthday problem):

With 5 people, probability of a shared birthday is only ~2.6% → very small.
With 10 people, it jumps to ~11% → still low, but not negligible.
With 20 people, already ~41%! That’s almost a coin toss.
With 23 people, it’s ~50% → this is the famous paradox result. Half the time, 23 people will have a match.
With 30 people, ~70% chance — more likely than not.
With 50 people, ~97% chance — basically guaranteed.'''

# --conclusion---
'''Our intuition usually underestimates the probability of coincidences. We think 23 people is “too small” for a birthday clash, 
but actually the number of possible pairs grows very quickly:
(23 n 2​)=253 pairs!
Each pair is a chance for a match, so the probability adds up fast.

This paradox teaches us to not always trust raw intuition in probability — 
super useful in data science when reasoning about collisions, hashing, random sampling, etc.'''