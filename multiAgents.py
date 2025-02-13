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


import logging
from util import manhattanDistance
from game import Directions, Actions
import random
import util
from game import Agent
from pacman import GameState
# from collections import deque

logging.basicConfig(level=logging.INFO, format='%(message)s')
called = 0  # Number of times something is called, used for debugging


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** YOUR CODE HERE ***"
        # logging.getLogger().setLevel(logging.DEBUG)

        # Counting the number of times this section is called. Useful sometimes.
        global called
        # if called == 10:
        #     util.pause()
        # if calledEvalFunction >= 10:
        #     util.pause()
        #     raise Exception("Called too many times %r", calledEvalFunction)
        called += 1
        logging.debug("\n\n\n----------\n\n\n")
        logging.debug("[ Eval Function Called %r ]", called)

        # Thoughts:
        # First of all, let's think of a plan for this evaluation function.
        # 1. The number one priority is to not get destroyed by a ghost next to us. Well, also don't intentionally run into a ghost too. So if a game state ends the game, it must be super highly penalized.
        # 2. While avoiding the ghost, it's still best to eat as many dots as possible. To make it simple, let's just make every pellet that is above pacman give 1 point to the decision of moving up.
        # 3. Not moving shouldn't really be rewarded, but it is still always an option. Don't forget coding it in.

        # Note:
        # According to PacmanRules in pacman.py
        # - Eating a dot gives 10 score
        # - Winning the game gives 500
        # - Eating the scared ghost gives 200
        # - Losing the game loses 500
        # - Each time step, the existence cost is 1
        # - Power pellets give no score

        value = 0.0   # The evaluated score of the action, higher numbers are better

        # Useful information you can extract from a GameState (pacman.py)
        # s is for successor
        sGameState = currentGameState.generatePacmanSuccessor(action)
        # logging.debug(f"currentGameState = {currentGameState}")
        sPosition = sGameState.getPacmanPosition()
        # logging.debug(f"pacmanPosition={pacmanPosition}")
        sFoodGrid = sGameState.getFood()
        sFoodList = sFoodGrid.asList()
        sGhostStates = sGameState.getGhostStates()
        # scaredTimes = [ghostState.scaredTimer for ghostState in sGhostStates]

        # This is a trap, intended to trap and pause the game in specific situations.
        # def trap():
        #     pass

        # trap()

        #
        # Those are functions that are mostly just to make the code look more tidy
        def avoidImmediateGhost():
            nonlocal value
            for ghostState in sGhostStates:
                distanceToGhost = util.manhattanDistance(sPosition, ghostState.getPosition())
                if distanceToGhost == 0:
                    value = -1000.0
                    break
                if distanceToGhost == 1:
                    value = -800.0
                    break

        def lookForFoodInActionDirection():
            nonlocal value
            VALUE_OF_FOOD = 1.0
            DECAY_PER_DISTANCE = 0.5

            for food in sFoodList:
                if (
                    (action == "North" and food[1] > sPosition[1]) or
                    (action == "South" and food[1] < sPosition[1]) or
                    (action == "East" and food[0] > sPosition[0]) or
                    (action == "West" and food[0] < sPosition[0])
                ):
                    foodValue = VALUE_OF_FOOD * (DECAY_PER_DISTANCE ** (
                        util.manhattanDistance(sPosition, food)))
                    logging.debug("foodValue: %r", foodValue)
                    value += foodValue

        lookForFoodInActionDirection()

        avoidImmediateGhost()

        value += sGameState.getScore()

        logging.getLogger().setLevel(logging.INFO)
        return value


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


# Helper functions used
def callFlagForQ5():
    # Counting the number of times this section is called. Useful sometimes.
    global called
    # if called == 10:
    #     util.pause()
    # if calledEvalFunction >= 10:
    #     util.pause()
    #     raise Exception("Called too many times %r", calledEvalFunction)
    called += 1

    util.pause()
    logging.debug("\n\n\n----------\n\n\n")
    logging.debug("[ Called %r ]", called)


def isBigger(a, b): return a > b


def isSmaller(a, b): return a < b


def getNextAgentData(agentIndex, depth, gameState):
    nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
    nextDepth = depth
    nextComparison = isSmaller
    if nextAgentIndex == 0:  # If it's Pacman's turn again
        nextDepth -= 1
        nextComparison = isBigger
    return nextAgentIndex, nextDepth, nextComparison


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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
        # logging.getLogger().setLevel(logging.DEBUG)

        def minimaxCore(agentIndex, depth, gameState, comparisonFunction):
            # Returns bestAction, and value of that action
            bestAction = None
            bestActionValue = None

            def endOfAnalysis():
                logging.debug(f"Max Depth Reached")
                evaluatedActionValue = self.evaluationFunction(gameState)
                logging.debug(f"evaluatedActionValue = {evaluatedActionValue}")
                return None, evaluatedActionValue

            # callFlag()

            logging.debug(f"agentIndex = {agentIndex}")
            logging.debug(f"depth = {depth}")
            logging.debug(f"comparisonFunction = {comparisonFunction}")

            # If we already reached the depth limit, no need to do anything else
            # Just return the evaluated value
            if depth == 0:
                logging.debug(f"Max Depth Reached")
                return endOfAnalysis()

            nextAgentIndex, nextDepth, nextComparison = getNextAgentData(agentIndex, depth, gameState)

            # Analyze all actions possible for pacman, and store the actions in a list
            actionList = gameState.getLegalActions(agentIndex)

            # If the actionList is empty, then we are at the end of the game tree
            # No more analysis is needed
            if len(actionList) == 0:
                logging.debug(f"End of Game Tree Reached")
                return endOfAnalysis()

            # If we need to recurse, recurse through every action in the list
            for action in actionList:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                logging.debug(f"successorGameState = {successorGameState}")

                # Do the recursion
                _, actionValue = minimaxCore(nextAgentIndex, nextDepth, successorGameState, nextComparison)

                if bestActionValue is None or comparisonFunction(actionValue, bestActionValue):
                    bestActionValue = actionValue
                    bestAction = action

            logging.debug(f"bestAction, bestActionValue = {bestAction, bestActionValue}")
            return bestAction, bestActionValue

        # The action we will return because we think it's the best
        returnAction = None

        # Not so temporary work around that simply just works (in theory)
        nextAgentIndex = 0
        nextDepth = self.depth
        nextComparison = isBigger
        returnAction, _ = minimaxCore(nextAgentIndex, nextDepth, gameState, nextComparison)

        logging.getLogger().setLevel(logging.INFO)
        return returnAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        "*** YOUR CODE HERE ***"
        # logging.getLogger().setLevel(logging.DEBUG)

        def alphabetaCore(agentIndex, depth, alpha, beta, gameState, comparisonFunction):
            # Returns bestAction, and value of that action
            bestAction = None
            bestActionValue = None

            def endOfAnalysis():
                evaluatedActionValue = self.evaluationFunction(gameState)
                logging.debug(f"evaluatedActionValue = {evaluatedActionValue}")
                return None, evaluatedActionValue

            # callFlag()

            logging.debug(f"agentIndex = {agentIndex}")
            logging.debug(f"depth = {depth}")
            logging.debug(f"comparisonFunction = {comparisonFunction}")

            # If we already reached the depth limit, no need to do anything else
            # Just return the evaluated value
            if depth == 0:
                logging.debug(f"Max Depth Reached")
                return endOfAnalysis()

            nextAgentIndex, nextDepth, nextComparison = getNextAgentData(agentIndex, depth, gameState)

            # Analyze all actions possible for pacman, and store the actions in a list
            actionList = gameState.getLegalActions(agentIndex)

            # If the actionList is empty, then we are at the end of the game tree
            # No more analysis is needed
            if len(actionList) == 0:
                logging.debug(f"End of Game Tree Reached")
                return endOfAnalysis()

            # If we need to recurse, recurse through every action in the list
            for action in actionList:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                logging.debug(f"successorGameState = {successorGameState}")

                # Do the recursion, but only assign it to v if it is more "comparisonFunction" than v
                # Note that v is bestActionValue
                _, actionValue = alphabetaCore(nextAgentIndex, nextDepth, alpha, beta, successorGameState, nextComparison)

                # v = max/min(v, value(successor, alpha, beta))
                if bestActionValue is None or comparisonFunction(actionValue, bestActionValue):
                    bestActionValue = actionValue
                    bestAction = action

                if comparisonFunction == isBigger:
                    # if v > beta return v
                    if bestActionValue > beta:
                        return bestAction, bestActionValue
                    # alpha = max(alpha, v)
                    alpha = max(alpha, bestActionValue)

                elif comparisonFunction == isSmaller:
                    # if v < alpha return v
                    if bestActionValue < alpha:
                        return bestAction, bestActionValue
                    # beta = min(beta, v)
                    beta = min(beta, bestActionValue)

                else:
                    raise Exception("Unexpected comparison function encountered. Expected either 'isBigger' or 'isSmaller', but got: {}".format(comparisonFunction))

            logging.debug(f"bestAction, bestActionValue = {bestAction, bestActionValue}")
            return bestAction, bestActionValue

        # The action we will return because we think it's the best
        returnAction = None

        # Not so temporary work around that simply just works (in theory)
        nextAgentIndex = 0
        nextDepth = self.depth
        nextComparison = isBigger
        alpha = float('-inf')
        beta = float('inf')
        returnAction, _ = alphabetaCore(nextAgentIndex, nextDepth, alpha, beta, gameState, nextComparison)

        logging.getLogger().setLevel(logging.INFO)
        return returnAction


def callFlagForQ5():
    # Counting the number of times this section is called. Useful sometimes.
    global called
    # if called == 10:
    #     util.pause()
    # if calledEvalFunction >= 10:
    #     util.pause()
    #     raise Exception("Called too many times %r", calledEvalFunction)
    called += 1

    util.pause()
    logging.debug("\n\n\n----------\n\n\n")
    logging.debug("[ Called %r ]", called)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # logging.getLogger().setLevel(logging.DEBUG)
        if PAUSE_EVERY_STEP:
            callFlagForQ5()

        def expectimaxCore(agentIndex, depth, gameState):
            # Returns action, and value of that action

            def endOfAnalysis():
                evaluatedActionValue = self.evaluationFunction(gameState)
                logging.debug(f"evaluatedActionValue = {evaluatedActionValue}")
                return None, evaluatedActionValue

            # callFlag()
            # logging.debug(f"depth = {depth}")

            # If we already reached the depth limit, no need to do anything else
            # Just return the evaluated value
            if depth == 0:
                logging.debug(f"Max Depth Reached")
                return endOfAnalysis()

            nextAgentIndex, nextDepth, _ = getNextAgentData(agentIndex, depth, gameState)

            # Analyze all actions possible for pacman, and store the actions in a list
            actionList = gameState.getLegalActions(agentIndex)

            # If the actionList is empty, then we are at the end of the game tree
            # No more analysis is needed
            if len(actionList) == 0:
                logging.debug(f"End of Game Tree Reached")
                return endOfAnalysis()

            # If we are the pacman
            if agentIndex == 0:
                bestAction = None
                bestActionValue = None

                # If we need to recurse, recurse through every action in the list
                for action in actionList:
                    successorGameState = gameState.generateSuccessor(agentIndex, action)
                    logging.debug(f"successorGameState = {successorGameState}")

                    # Do the recursion
                    # Note that v is bestActionValue
                    _, actionValue = expectimaxCore(nextAgentIndex, nextDepth, successorGameState)

                    # v = max(v, value(successor))
                    if bestActionValue is None or (actionValue > bestActionValue):
                        bestActionValue = actionValue
                        bestAction = action

                logging.debug(f"bestAction, bestActionValue = {bestAction, bestActionValue}")
                return bestAction, bestActionValue

            # If we are stupid ghosts
            else:
                actionValueList = []
                for action in actionList:
                    # Get the average value, recursion
                    successorGameState = gameState.generateSuccessor(agentIndex, action)
                    logging.debug(f"successorGameState = {successorGameState}")

                    _, actionValue = expectimaxCore(nextAgentIndex, nextDepth, successorGameState)

                    actionValueList.append(actionValue)
                averageActionValue = sum(actionValueList) / len(actionValueList)
                return None, averageActionValue

        # The action we will return because we think it's the best
        returnAction = None

        # Not so temporary work around that simply just works (in theory)
        nextAgentIndex = 0
        nextDepth = self.depth
        returnAction, _ = expectimaxCore(nextAgentIndex, nextDepth, gameState)

        logging.getLogger().setLevel(logging.INFO)
        return returnAction


# positionsVisitedCount = {}

# Initialize a deque to store the last 5 positions
# lastPositions = deque(maxlen=5)


# def recordPosition(position):
#     lastPositions.append(position)


# def getLastPositions():
#     return list(lastPositions)


def betterEvaluationFunction(gameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function (question 5).

    DESCRIPTION:

    """
    "*** YOUR CODE HERE ***"
    if VERBOSE:
        logging.getLogger().setLevel(logging.DEBUG)

    # First, let's try to understand the question.
    # In this final exam / benchmark, the ExpectimaxAgent will be called, and we will be using this evaluation function when the Agent needs an estimate value for the state.
    # The ghost is a pure random ghost, and... I assume the default depth is 2?
    # Yup, depth 2 is the default depth. So yeah.
    # Find a way to make the evaluation function accurately predict how good a state is.

    # Note:
    # According to PacmanRules in pacman.py
    # - Eating a dot gives 10 score
    # - Winning the game gives 500
    # - Eating the scared ghost gives 200
    # - Losing the game loses 500
    # - Each time step, the existence cost is 1
    # - Power pellets give no score

    position = gameState.getPacmanPosition()
    foodList = gameState.getFood().asList()

    def aStarDistance(start, goal):
        # A fucking entire a star path search to calculate "distance" accurately
        from util import PriorityQueue
        walls = gameState.getWalls()
        frontier = PriorityQueue()
        frontier.push((start, 0), 0)
        visited = {}

        while not frontier.isEmpty():
            current, cost = frontier.pop()
            if current in visited:
                continue
            visited[current] = cost

            if current == goal:
                return cost

            x, y = current
            for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                dx, dy = Actions.directionToVector(direction)
                nextx = int(x + dx)
                nexty = int(y + dy)
                if 0 <= nextx < walls.width and 0 <= nexty < walls.height and not walls[nextx][nexty]:
                    nextPos = (nextx, nexty)
                    new_cost = cost + 1
                    if nextPos not in visited or new_cost < visited.get(nextPos, float('inf')):
                        priority = new_cost + manhattanDistance(nextPos, goal)
                        frontier.push((nextPos, new_cost), priority)

        return float('inf')  # No path found

    def penaltyFromCloseToGhost():
        # Losing is the main reason we score low. Let's avoid that.
        DANGER_DISTANCE = 3
        DANGER_PENALTY = 500
        distanceToClosestGhost = min([manhattanDistance(gameState.getPacmanPosition(), ghost.getPosition()) for ghost in gameState.getGhostStates()])
        logging.debug(f"distanceToClosestGhost = {distanceToClosestGhost}")
        ghostPenalty = -(max(0, DANGER_DISTANCE - distanceToClosestGhost)) * (DANGER_PENALTY / DANGER_DISTANCE)
        return ghostPenalty

    def valueFromFoodList():
        # Now, let's make pacman be attracted to dots
        # In fact, how about make every dot attract pacman
        # This almost works now, I'm gonna do an actual fucking path find
        VALUE_OF_FOOD = 10.0
        DECAY_PER_DISTANCE = 0.05
        totalFoodValue = 0
        for food in foodList:
            foodValue = VALUE_OF_FOOD * (DECAY_PER_DISTANCE ** (aStarDistance(position, food)))
            # logging.debug(f"food = {food}")
            # logging.debug(f"foodValue = {foodValue}")
            totalFoodValue += foodValue
        return totalFoodValue

    # The main problem is pacman staying still
    # Gonna add a penalty for states we've already visited
    # It's gonna be a really small penalty
    # global positionsVisitedCount
    # if position in positionsVisitedCount:
    #     positionsVisitedCount[position] += 1
    # else:
    #     positionsVisitedCount[position] = 1
    # lazyPenalty = -positionsVisitedCount[position] * 0.1
    # logging.debug(f"position = {position}")
    # logging.debug(f"lazyPenalty = {lazyPenalty}")
    # value += lazyPenalty

    # That didn't work, try to only penalize the last 5 locations visited
    # LAZY_PENALTY = 100
    # totalLazyPenalty = 0
    # for lastPosition in getLastPositions():
    #     if lastPosition == position:
    #         totalLazyPenalty += LAZY_PENALTY
    # logging.debug(f"position = {position}")
    # logging.debug(f"totalLazyPenalty = {totalLazyPenalty}")
    # value -= totalLazyPenalty
    # # Ok, so the problem is, this is called every time we try to analyze the game state
    # # Which is actually the successor
    # recordPosition(position)

    def totalValue():
        totalValue = 0

        totalValue += gameState.getScore()

        ghostPenalty = penaltyFromCloseToGhost()
        logging.debug(f"ghostPenalty = {ghostPenalty}")
        totalValue += ghostPenalty

        totalFoodValue = valueFromFoodList()
        logging.debug(f"totalFoodValue = {totalFoodValue}")
        totalValue += totalFoodValue

        return totalValue

    value = totalValue()

    logging.getLogger().setLevel(logging.INFO)
    return value


# Abbreviation
better = betterEvaluationFunction
PAUSE_EVERY_STEP = False
VERBOSE = False


# PAUSE_EVERY_STEP = True
# VERBOSE = True
