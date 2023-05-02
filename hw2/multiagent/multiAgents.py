# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # return successorGameState.getScore()
        newFoodList = newFood.asList()
        if len(newFoodList) == 0:
            return float("inf")
        minFoodDist = min([manhattanDistance(newPos, food) for food in newFoodList])
        minGhostDist = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
        if minGhostDist == 0:
            return float("-inf")
        return successorGameState.getScore() + 1.0/minFoodDist

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return self.minimax(gameState, 0, 0)[1]

    def minimax(self, gameState, agentIndex, depth):
        if depth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None
        if agentIndex == 0:
            return self.get_max_value(gameState, agentIndex, depth)
        else:
            return self.get_min_value(gameState, agentIndex, depth)

    def get_max_value(self, gameState, agentIndex, depth):
        max_value = float("-inf")
        max_action = None
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            value = self.minimax(successor, (agentIndex + 1) % gameState.getNumAgents(), depth + 1)[0]
            if value > max_value:
                max_value = value
                max_action = action
        return max_value, max_action

    def get_min_value(self, gameState, agentIndex, depth):
        min_value = float("inf")
        min_action = None
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            value = self.minimax(successor, (agentIndex + 1) % gameState.getNumAgents(), depth + 1)[0]
            if value < min_value:
                min_value = value
                min_action = action
        return min_value, min_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return self.alpha_beta(gameState, 0, 0, float("-inf"), float("inf"))[1]

    def alpha_beta(self, gameState, agentIndex, depth, alpha, beta):
        if depth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None
        if agentIndex == 0:
            return self.get_max_value(gameState, agentIndex, depth, alpha, beta)
        else:
            return self.get_min_value(gameState, agentIndex, depth, alpha, beta)

    def get_max_value(self, gameState, agentIndex, depth, alpha, beta):
        max_value = float("-inf")
        max_action = None
        for action in gameState.getLegalActions(agentIndex):
            # if action == Directions.STOP:
            #     continue
            successor = gameState.generateSuccessor(agentIndex, action)
            value = self.alpha_beta(successor, (agentIndex + 1) % gameState.getNumAgents(), depth + 1, alpha, beta)[0]
            if value > max_value:
                max_value = value
                max_action = action
            if max_value > beta:
                return max_value, max_action
            alpha = max(alpha, max_value)
        return max_value, max_action

    def get_min_value(self, gameState, agentIndex, depth, alpha, beta):
        min_value = float("inf")
        min_action = None
        for action in gameState.getLegalActions(agentIndex):
            # if action == Directions.STOP:
            #     continue
            successor = gameState.generateSuccessor(agentIndex, action)
            value = self.alpha_beta(successor, (agentIndex + 1) % gameState.getNumAgents(), depth + 1, alpha, beta)[0]
            if value < min_value:
                min_value = value
                min_action = action
            if min_value < alpha:
                return min_value, min_action
            beta = min(beta, min_value)
        return min_value, min_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return self.Expectimax(gameState, 0, 0)[1]

    def Expectimax(self, gameState, agentIndex, depth):
        if depth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None
        if agentIndex == 0:
            return self.get_max_value(gameState, agentIndex, depth)
        else:
            return self.get_expect_value(gameState, agentIndex, depth)

    def get_max_value(self, gameState, agentIndex, depth):
        max_value = float("-inf")
        max_action = None
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            value = self.Expectimax(successor, (agentIndex + 1) % gameState.getNumAgents(), depth + 1)[0]
            if value > max_value:
                max_value = value
                max_action = action
        return max_value, max_action

    def get_expect_value(self, gameState, agentIndex, depth):
        expect_value = 0
        expect_action = None
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            value = self.Expectimax(successor, (agentIndex + 1) % gameState.getNumAgents(), depth + 1)[0]
            expect_value += value
        return expect_value / len(gameState.getLegalActions(agentIndex)), expect_action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    pacman_position = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()
    ghost_states = currentGameState.getGhostStates()
    capsule_list = currentGameState.getCapsules()
    score = currentGameState.getScore()

    if len(food_list) == 0:
        return score + 1000
    else:
        food_count = len(food_list)

    if len(capsule_list) == 0:
        return score
    else:
        capsule_count = len(capsule_list)

    nearest_food = float("inf")
    nearest_food = min([manhattanDistance(pacman_position, food) for food in food_list])

    nearest_ghost = float("inf")
    nearest_ghost = min([manhattanDistance(pacman_position, ghost.getPosition()) for ghost in ghost_states])

    nearest_capsule = float("inf")
    nearest_capsule = min([manhattanDistance(pacman_position, capsule) for capsule in capsule_list])

    if currentGameState.isWin():
        return float("inf")

    if currentGameState.isLose():
        return float("-inf")

    if nearest_ghost == 0:
        return float("-inf")

    return score + 100.0 / nearest_food + 10.0 / nearest_capsule - 100000.0 * food_count - 10.0 * capsule_count
# Abbreviation
better = betterEvaluationFunction
