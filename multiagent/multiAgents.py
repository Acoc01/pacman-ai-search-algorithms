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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        #print(successorGameState.getFood().asList())

        #Calculo la distancia manhattan para que pacman siga la comida mas cercana y escape de
        #los fantasmas cercanos.
        newFood = successorGameState.getFood().asList()
        nearestFood = float('inf');
        for food in newFood:
          nearestFood = min(nearestFood, manhattanDistance(food, newPos))
        ghostPos = successorGameState.getGhostPositions()
        for ghost in ghostPos:
          #print(ghost)
          if(manhattanDistance(newPos, ghost) < 2):
              return -float('inf')
        #print(1.0/nearestFood+successorGameState.getScore(),' ',successorGameState.getScore())
        return successorGameState.getScore() + 2.0/nearestFood;

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
        """
        "*** YOUR CODE HERE ***"
        #Obtenemos acciones del pacman
        pacmanActions = gameState.getLegalActions(0)
        # for action in pacmanActions:
        #     values.append(self.minimax(gameState.generateSuccessor(0,action), 1))
        # Generamos una lista con los valores de las acciones del pacman y nos quedamos con la mejor
        return max(pacmanActions, key=lambda x: self.minimax(gameState.generateSuccessor(0, x),1))
    def minimax(self, gameState, turn):
        #Numero de agentes
        agents = gameState.getNumAgents()
        #Profundidad en base a turnos
        depth = turn // agents
        #Arreglo circular
        index = turn % agents
        #Condicion de termino
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        #Acciones de agente actual
        agentActions = gameState.getLegalActions(index)
        #Lista de la evaluacion de tomar la accion x
        values = []
        for action in agentActions:
          values.append(self.minimax(gameState.generateSuccessor(index, action), turn + 1))
        #Si soy pacman, maximizo
        if index == 0:
            return max(values)
        #Si soy fantasma minimizo
        else:
            return min(values)
        #util.raiseNotDefined()

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
        #Acciones de pacman
        actions = gameState.getLegalActions(0)
        #Definimos alpha y beta
        alpha = -float('inf')
        beta = float('inf')
        #Lista para el valor de las acciones del pacman
        values = []
        #Para cada una de sus acciones evaluamos, actualizamos el alpha para poder evaluar
        # en la siguiente iteracion y almacenamos la evaluacion actual
        for action in actions:
            val = self.alphaBeta(gameState.generateSuccessor(0,action), 1, alpha, beta)
            alpha = max(alpha,val)
            values.append(val)
        #Luego, buscamos la mejor accion utilizando el valor alpha como referencia
        for i in range(len(actions)):
            if alpha == values[i]:
              return actions[i]
      
    def alphaBeta(self, gameState, turn, alpha, beta):
        #Numero de agentes
        agents = gameState.getNumAgents()
        #Profundidad en base al numero de agentes y el turno
        depth = turn // agents
        #Arreglo circular
        index = turn % agents
        #Condicion de termino
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        #Acciones de agente actual
        agentActions = gameState.getLegalActions(index)
        #Definimos alpha o beta en cada caso
        if index == 0:
            agentVal = -float('inf')
        else:
            agentVal = float('inf')
        #Si soy pacman, maximizo
        if index == 0:
            for action in agentActions:
                agentVal = max(self.alphaBeta(gameState.generateSuccessor(index, action), turn + 1, alpha, beta),agentVal)
                if agentVal > beta: return agentVal
                else: alpha = max(agentVal, alpha)
        #Si soy fantasma, minimizo
        else:
            for action in agentActions:
                agentVal = min(self.alphaBeta(gameState.generateSuccessor(index, action), turn + 1, alpha, beta),agentVal)
                if agentVal < alpha: return agentVal
                else: beta = min(agentVal, beta)
        return agentVal

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
        pacmanActions = gameState.getLegalActions(0)
        # for action in pacmanActions:
        #     values.append(self.minimax(gameState.generateSuccessor(0,action), 1))

        #Evaluo una lista, por cada accion se genera un valor y me quedo con el maximo
        return max(pacmanActions, key=lambda x: self.minimax(gameState.generateSuccessor(0, x),1))
    def minimax(self, gameState, turn):
        #Numero total de agentes
        agents = gameState.getNumAgents()
        #Profundidad en base a un turno, si hay n agentes, n // n = 1
        depth = turn // agents
        #Arreglo circular
        index = turn % agents
        #Condicion de termino
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        #Obtengo las acciones del agente actual
        agentActions = gameState.getLegalActions(index)
        #Lista para valores de cada una de sus acciones
        values = []
        #Se evalua cada una de sus acciones
        for action in agentActions:
          values.append(self.minimax(gameState.generateSuccessor(index, action), turn + 1))
        #Si soy pacman, me interesa maximizar el valor de mis acciones
        if index == 0:
            return max(values)
        #Si soy fantasma, me interesa minimizar, en este caso devolvemos el promedio por como funciona el algoritmo.
        else:
            return sum(values) / len(values)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    ##Intento replicar lo mismo que en mi reflex agent
    if currentGameState.isLose(): return -float('inf')
    if currentGameState.isWin(): return float('inf')
    food = currentGameState.getFood().asList()
    position = currentGameState.getPacmanPosition()
    ghostPosition = currentGameState.getGhostPositions()

    newFood = currentGameState.getFood().asList()
    nearestFood = float('inf');
    ghostStates = currentGameState.getGhostStates()
    for food in newFood:
      nearestFood = min(nearestFood, manhattanDistance(food, position))
    ghostTooClose = -1.0
    for g in ghostPosition:
        if(manhattanDistance(position, g) < 4):
          ghostTooClose += manhattanDistance(position,g)
    if(ghostTooClose == -1.0):
        ghostTooClose = 1.0
    print("nearest food: ",2.0/nearestFood)
    print("ghosts position: ", 1.0*ghostTooClose)
    return currentGameState.getScore() + 2.0/nearestFood +0.4 - ghostTooClose * 1.0




# Abbreviation
better = betterEvaluationFunction

