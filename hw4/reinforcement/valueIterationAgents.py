# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        while self.iterations > 0:
            temps = self.values.copy()
            for state in self.mdp.getStates():
                allActions = self.mdp.getPossibleActions(state)
                possibleValues = []
                for action in allActions:
                    endStates = self.mdp.getTransitionStatesAndProbs(state, action)
                    qValue = 0
                    for nextState, prob in endStates:
                        qValue += prob * (self.mdp.getReward(state, action, nextState) + self.discount * temps[nextState])
                    possibleValues.append(qValue)
                if len(possibleValues) == 0:
                    self.values[state] = 0
                else:
                    self.values[state] = max(possibleValues)
            self.iterations -= 1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        allstates = self.mdp.getTransitionStatesAndProbs(state, action)
        qValue = 0
        for nextState, prob in allstates:
            qValue += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        finalAction = ""
        maximum = float("-inf")
        for action in self.mdp.getPossibleActions(state):
            qValue = self.computeQValueFromValues(state, action)
            if qValue > maximum:
                maximum = qValue
                finalAction = action
        return finalAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            state = self.mdp.getStates()[i % len(self.mdp.getStates())]
            answer = self.computeActionFromValues(state)
            if answer is not None:
                self.values[state] = self.computeQValueFromValues(state, answer)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def all_q_values(self, state):
        allActions = self.mdp.getPossibleActions(state)
        q_values = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            q_values[action] = self.computeQValueFromValues(state, action)

        return q_values

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {}
        for state in self.mdp.getStates():
            predecessors[state] = set()
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        predecessors[nextState].add(state)
        pq = util.PriorityQueue()
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                diff = abs(self.values[state] - max(self.all_q_values(state).values()))
                pq.update(state, -diff)
        for i in range(self.iterations):
            if pq.isEmpty():
                return
            state = pq.pop()
            if not self.mdp.isTerminal(state):
                self.values[state] = max(self.all_q_values(state).values())
            for p in predecessors[state]:
                if not self.mdp.isTerminal(p):
                    diff = abs(self.values[p] - max(self.all_q_values(p).values()))
                    if diff > self.theta:
                        pq.update(p, -diff)
