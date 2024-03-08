# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.values = util.Counter()

    # QUESTION 5 (Q-Learning)
    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.values[state, action]

    # QUESTION 5 (Q-Learning)
    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        
        # terminal state
        q_max = 0.0

        # in cazul in care exista actiuni returnam q-value
        if len(actions) is not 0:
          best_action = self.getPolicy(state)
          q_max = self.getQValue(state, best_action)

        # functia va returna 0 in cazul in care ne aflam in terminal state (nu exista actiuni disponibile)  
        return q_max

    # QUESTION 5 (Q-Learning)
    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        
        # terminal state
        best_action = None 

        if len(actions) is not 0:
          q_values = util.Counter()
          
          # populam tabelul cu q-value pentru fiecare actiune
          for a in actions:
              q_values[a] = self.getQValue(state, a)
          
          # in final alegem cea mai buna actiune folosind argmax
          best_action = q_values.argMax()

        # daca ne aflam in terminal state (nu mai avem actiuni) functia va returna none
        return best_action
    
    # QUESTION 6 (Epsilon Greedy)
    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        actions = self.getLegalActions(state)

        # terminal test
        action = None
        "*** YOUR CODE HERE ***"
        if len(actions) is not 0:
          # folosim hint-urile pentru a ajunge la epsilon-greedy action selection
          if (flipCoin(self.epsilon)):    
            action = random.choice(actions)
          else:
            action = self.getPolicy(state)
        # in cazul in care nu mai avem actiuni vom returna none
        return action

    # QUESTION 5 (Q-Learning)
    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # formula din suportul de curs
        # Q(s,a)←(1−α)Q(s,a)
        Q = (1 - self.alpha) * self.getQValue(state, action)
        # +α[r+γmaxa′ Q(s′,a′)]
        Q += self.alpha * (reward + (self.discount * self.getValue(nextState))) 
          
        self.values[state, action] = Q

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

# QUESTION 7 (PACMAN)
class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

# QUESTION 9 (Approximate Q-Learning)
class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights
    
    # QUESTION 9 (Approximate Q-Learning)
    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        if action is None:
            return 0.0
        
        # ambele sunt counters
        features = self.featExtractor.getFeatures(state, action)
        weights = self.getWeights()

        # initializam suma
        weighted_sum = 0

        # calculam media ponderata a tuturor feature-urilor
        for f in features:
            weighted_sum += weights[f] * features[f]
        return weighted_sum
    
    # QUESTION 9 (Approximate Q-Learning)
    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)

        # am folosit formulele prezentate aici https://youtu.be/XafrqwHfBKE?si=-XG8wAdevZHHcOOq&t=2758
        difference = (reward + (self.discount * (self.getValue(nextState)))) - self.getQValue(state, action)

        for f in features:
            self.weights[f] += self.alpha * difference * features[f]

        return self.weights

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
