import random
import math

# draw: [float] -> int
# pick an index from the given list of floats proportionally
# to the size of the entry (i.e. normalize to a probability
# distribution and draw according to the probabilities).
def draw(weights):
    #print weights
    #Issue: Sometimes the "weights" list is holding a nan element causing this routine to break.
    choice = random.uniform(0, sum(weights))
    choiceIndex = 0

    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choiceIndex

        choiceIndex += 1


#Add sanity checking on weights too
def Bubeck_draw(weights, t=1):

    choice = random.uniform(0, sum(weights))
    choiceIndex = 0

    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choiceIndex

        choiceIndex += 1

# distr: [float] -> (float)
# Normalize a list of floats to a probability distribution.  Gamma is an
# egalitarianism factor, which tempers the distribution toward being uniform as
# it grows from zero to one.
def distr(weights, gamma=0.0, rnd=1):
    theSum = float(sum(weights))
    if rnd % 100000 == 0:
        for itr in range(len(weights)):
            weights[itr] /= theSum

    theSum = float(sum(weights))
    return tuple((1.0 - gamma) * (w / theSum) + (gamma / len(weights)) for w in weights)

def Bubeck_distr(weights):
    theSum = float(sum(weights))
    return tuple(w / theSum for w in weights)

def exp3S_distr(weights, gamma=1.0):
    theSum = float(sum(weights))
    return tuple((1.0 - gamma) * (w / theSum) + (gamma / len(weights)) for w in weights)


def mean(aList):
   theSum = 0
   count = 0

   for x in aList:
      theSum += x
      count += 1

   return 0 if count == 0 else theSum / count
