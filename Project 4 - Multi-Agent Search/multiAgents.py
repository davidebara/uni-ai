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
from pacman import GameState

# 1. REFLEX AGENT
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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # Mai intai incercam sa printam variabilele
        """Ne afiseaza urmatoarea stare a jocului, unde PACMAN e >, strigoiul G si fructul o"""
        # print("succesorGameState:")
        # print(successorGameState)

        """Ne afiseaza pozitia lui PACMAN sub forma a doua coordonate x,y"""
        # print("newPos:")
        # print(newPos)

        """Ne afiseaza unde se afla fructele, pare ca T sunt fructe (true) si F sunt locuri fara fructe (false)"""
        # print("newFood:")
        # print(newFood.asList())

        """Ne afiseaza un array ce are ca elemente strigoii, fiecare obiect avand o adresa hexadecimala"""
        # print(f"newGhostStates: {newGhostStates}")

        """Ne afiseaza un array cu secundele pentru care strigoii sunt speriati de PACMAN, din momentul in care acesta mananca un fruct mare"""
        # print(f"newScaredTimes: {newScaredTimes}")
        # print("==========================================")

        evaluation = currentGameState.getScore()
        max_num = float('inf')

        # CASTIG
        """
        - daca urmatorul gamestate este un castig pentru PACMAN, vom returna o valoare foarte mare pentru a-l incuraja pe PACMAN sa se indrepte in acea directie.
        """
        if(successorGameState.isWin()):
            return max_num

        # DISTANTE FRUCTE
        """
        - pornim de la premisa ca nu exista fructe iar pe masura ce parcurgem lista acestora stabilim care este cel mai apropiat fruct
        - ne folosim de reciproca pentru a favoriza valorile fructele mai apropiate
        """
        f = max_num
        for food in newFood.asList():
            val = manhattanDistance(food, newPos)
            f = min(f, val)
        evaluation += 1/f * 10

        # NUMAR FRUCTE
        """
        - obiectivul nostru este sa il facem pe PACMAN sa manance toate fructele din labirint
        - pentru a atinge acest obiectiv vom creste scorul de fiecare data cand PACMAN mananca fructe, adica in urmatorul game state sunt mai putine fructe decat in cel prezent
        """
        f_current = currentGameState.getNumFood()
        f_successor = successorGameState.getNumFood()
        if(f_successor < f_current):
            evaluation += 150

        # DISTANTE FRUCTE MARI
        """
        - il incurajam pe pacman sa urmareasca culeaga si fructele mari
        - acestea nu sunt esentiale pentru victoria lui pacman, deci folosim un coeficient mai mic
        """
        c = max_num
        for capsule in successorGameState.getCapsules():
            val = manhattanDistance(capsule, newPos)
            c = min(c,val)
        evaluation += 1/c * 0.7

        # NUMAR FRUCTE MARI
        """
        - similar cu schimbarea numarului de fructe normale
        """
        c_current = currentGameState.getCapsules()
        c_successor = successorGameState.getCapsules()
        if(len(c_current) > len(c_successor)):
            evaluation += 100

        # DISTANTE STRIGOI
        """
        - similar cu modul in care ne uitam la distantele pana la fructe vom aborda si situatia strigoilor
        - de data aceasta insa il vom descuraja pe PACMAN, scazand scorul sau de fiecare data cand se apropie de un strigoi
        """
        g = max_num
        for ghost in currentGameState.getGhostPositions():
            val = manhattanDistance(ghost, newPos)
            g = min(g,val)
        
        if(g == 0):
            evaluation -= max_num
        else:
            if g < max_num:
                evaluation -= 1/g * 10

        
        # OPRIREA PE LOC
        """
        - am observat ca PACMAN are momente in care se opreste in labirint, chiar daca ar putea culege fructe fara probleme. Pentru a preveni acest comportament vom scadea 50 din scorul lui PACMAN de fiecare data cand se opreste.
        - testand codul cu diverse valori am observat ca la valoarea 500 PACMAN incepe sa se deplaseze sus-jos in continuu, pana cand un strigoi se apropie de el.
        """
        if action == Directions.STOP:
            evaluation -= 1

        return evaluation

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

# 2. MINIMAX
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

        # NOTES (PSEUDOCOD CURS & PROJECT FILES)
        """
        - value(state) apeleaza max_value(state) sau min_value(state) in functie de tipul nodului pe care se afla
        - value(succesor) se refera la urmatorul nod, adica la o actiune (acum lucram cu actiuni, nu cu perechi stare-actiune)
        - ne putem folosi de getLegalActions(0) pentru a descoperi actiunile lui PACMAN (index = 1 sau 2 pentru strigoi)
        - ne putem folosi de generateSuccesor pentru a descoperi nodurile arborelui (ia 2 parametri, agentIndex and action)
        """
        actions = gameState.getLegalActions(0)
    
        # pornim de la un nod max (indicatie)
        # functioneaza ca si max_value, insa acum returnam o actiune
        if not actions:
            return None

        best_score = float('-inf')
        best_action = None

        for action in actions:
            val = self.value(gameState.generateSuccessor(0, action), self.depth, 1)
            if val > best_score:
                best_score = val
                best_action = action
        return best_action

    def value(self, gameState, depth, index):
        # noduri frunza (terminal state in curs)
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        
        # PACMAN (agent is MAX in curs)
        elif index == 0:
            return self.max_value(gameState, depth, index)
        
        # Strigoi (agent is MIN in curs)
        else:
            return self.min_value(gameState, depth, index)
        
    def max_value(self, gameState, depth, index):
        # nodurile de sub nodul max pe care ne aflam
        actions = gameState.getLegalActions(index)
        
        # daca nu mai exista noduri ne aflam pe o frunza
        if not actions:
            return self.evaluationFunction(gameState)

        # initializam v cu - infinit
        v = float('-inf')

        # in mod evident, value(succesor) va fi value(1) daca ne aflam pe nodul lui PACMAN (in material ni se spune ca arborele are urmatoarea structura: nivel max, 2 nivele min, nivel max etc.)
        for action in actions:
            # obtinem value(successor)
            val = self.value(gameState.generateSuccessor(index, action), depth, 1)
            # comparam v cu value of successor
            v = max(v, val)
        return v

    def min_value(self, gameState, depth, index):
        # nodurile de sub nodul min pe care ne aflam
        actions = gameState.getLegalActions(index)
        
        # daca nu mai exista noduri ne aflam pe o frunza
        if not actions:
            return self.evaluationFunction(gameState)

        # formula pentru a obtine indexul urmatorului agent indiferent de agentul curent (am generalizat formula asa cum ni se spunea in indicatie, astfel incat sa functioneze pentru mai multi strigoi)
        nextIndex = (index + 1) % gameState.getNumAgents()

        # daca urmatorul agent e PACMAN ne mutam in arbore pe nodul sau
        if nextIndex == 0:
            depth -= 1

        # initializam v cu + infinit
        v = float('inf')
        
        # for each successor of state ... (curs)
        for action in actions:
            # obtinem value(successor)
            val = self.value(gameState.generateSuccessor(index, action), depth, nextIndex)
            
            # comparam v cu value(successor)
            v = min(v,val)
        return v

# 3. ALPHA BETA
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # MAX's best option on path to root (curs)
        # se initializeaza cu o valoare mica, ce se va modifica in urma compararii cu nodurile copii
        alpha = float('-inf')

        # MIN's best option on path to root (curs)
        # se initializeaza cu o valoare mare, ce se va modifica in urma compararii cu nodurile copii
        beta = float('inf')

        # pornim de la un nod max (indicatie)
        # functioneaza ca si max_value dar de data aceasta returnam actiunea
        actions = gameState.getLegalActions(0)

        if not actions:
            return None

        best_score = float('-inf')
        best_action = None

        for action in actions:
            val = self.value(gameState.generateSuccessor(0, action), self.depth, alpha, beta, 1)
            
            if val > best_score:
                best_score = val
                best_action = action
            
            if best_score > beta:
                return best_action
            
            alpha = max(alpha, best_score)
        return best_action

    # noduri max (pacman)
    def max_value(self, gameState, depth, alpha, beta, index):
        # initializam v cu - infinit (curs)
        v = float('-inf')

        # obtinem successors of state
        actions = gameState.getLegalActions(index)
        
        # gestionam cazul in care nu mai avem succesori si ne aflam pe un nod frunza
        if not actions:
            return self.evaluationFunction(gameState)

        # for each successor of state (curs)
        for action in actions:
            # obtinem value(successor, 𝛼, β)
            val = self.value(gameState.generateSuccessor(index, action), depth, alpha, beta, 1)
            
            # comparam v cu value(successor, 𝛼, β)
            v = max(v, val)

            # in cazul in care valoarea nodului e mai mare decat beta (cea mai buna varianta a strigoiului) nu mai exploram nodurile invecinate si facem pruning (returnam v) (curs)
            # in conditiile noastre (presupunem ca agentii min sunt rationali), strigoiul nu ar alege niciodata o varianta mai mare decat o varianta deja proasta pentru el, deci ne putem opri
            if v > beta:
                return v
            
            # altfel cea mai buna varianta pentru pacman devine nodul v (curs)
            alpha = max(alpha, v)
        return v

    # noduri min (strigoi)
    def min_value(self, gameState, depth, alpha, beta, index):
        # obtinem, la fel ca la minimax, indexul urmatorului jucator
        nextIndex = (index + 1) % gameState.getNumAgents()
        
        # daca pacman e urmatorul jucator urcam in arbore
        if nextIndex == 0:
            depth -= 1
        
        # initializam v cu infinit (curs)
        v = float('inf')
        actions = gameState.getLegalActions(index)

        # gestionam cazul in care nu mai avem succesori si ne aflam pe un nod frunza
        if not actions:
            return self.evaluationFunction(gameState)

        # for each successor of state (curs)
        for action in actions:
            # obtinem value(successor, 𝛼, β)
            val = self.value(gameState.generateSuccessor(index, action), depth, alpha, beta, nextIndex)

            # comparam v cu value(successor, 𝛼, β)
            v = min(v, val)

            # in cazul in care valoarea nodului e mai mica decat alpha (cea mai buna varianta a lui pacman) nu mai exploram nodurile invecinate si facem pruning (returnam v) (curs)
            # pacman nu ar alege o varianta si mai mica decat o varianta deja proasta, deci ne putem opri
            if v < alpha:
                return v
            
            # altfel cea mai buna varianta a strigoiului devine v
            beta = min(beta, v)
        return v

    # value functioneaza ca si functia value de la minimax, aceasta primind in plus ca parametri valorile alpha si beta
    def value(self, gameState, depth, alpha, beta, index):
        # nod frunza
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        # nod pacman (max)
        elif index == 0:
            return self.max_value(gameState, depth, alpha, beta, index)
        # nod strigoi (min)
        else:
            return self.min_value(gameState, depth, alpha, beta, index)

# 4. EXPECTIMAX
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
        
        # pornim de la un nod max (indicatie)
        # functioneaza ca si max_value, insa acum returnam o actiune
        actions = gameState.getLegalActions(0)

        if not actions:
            return None
        
        best_score = float('-inf')
        best_action = None

        for action in actions:
            v = self.value(gameState.generateSuccessor(0, action), self.depth, 1)

            if v > best_score:
                best_score = v
                best_action = action
        return best_action

    # noduri max (pacman)
    def max_value(self, gameState, depth, index):
        # initializam v
        v = float('-inf')
        actions = gameState.getLegalActions(index)
        
        # gestionam cazul in care ne aflam pe un nod frunza
        if not actions:
            return self.evaluationFunction(gameState)

        # for each successor of state ... (curs)
        for action in actions:
            # value(successor)
            val = self.value(gameState.generateSuccessor(index, action), depth, 1)
            
            # v = max(v, value(successor))
            v = max(v, val)
        
        return v

    # noduri sansa (strigoi)
    def exp_value(self, gameState, depth, index):
        actions = gameState.getLegalActions(index)
        nextIndex = (index + 1) % gameState.getNumAgents()

        # daca urmatorul agent e pacman, urcam in arbore
        if nextIndex == 0:
            depth -= 1
        
        # intializam v = 0
        v = 0

        # for each successor of state ... (curs)
        for action in actions:
            # folosim aceeasi probabilitate pentru toate nodurile
            # inmultim cu 1.0 pentru a nu avea o impartire intreaga
            p = 1.0 / len (actions) * 1.0    
            val = self.value(gameState.generateSuccessor(index, action), depth, nextIndex) 
            v += p * val
        return v

    def value(self, gameState, depth, index):
        # nod frunza
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        # nod pacman (max)
        elif index == 0:
            return self.max_value(gameState, depth, index)
        # nod strigoi (sansa)
        else:
            return self.exp_value(gameState, depth, index)

# 5. EVALUATION FUNCTION
def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did> 
    """
    "*** YOUR CODE HERE ***"
    max_num = float('inf')
    position = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    evaluation = currentGameState.getScore()
    

    # distanta pana la fructe
    """
    - obiectivul lui pacman este sa manance toate fructele
    - astfel, calculam distanta pana la cel mai apropiat fruct
    - daca nu se gaseste niciun fruct inseamna ca am terminat jocul
    - la final adaugam inversul valorii pentru a favoriza fructele apropiate
    - daca puteam evalua si actiuni s-ar fi putut elimina opririle lui pacman cand strigoii sunt departe de el
    """
    f = max_num
    if foods:
        for food in foods:
            val = manhattanDistance(position, food)
            f = min(f, val)    
    evaluation += 1.0 / f * 10


    # distanta pana la fructe mari
    """
    - procedam la fel ca la fructele normale dar acordam o importanta mai mica fructelor mari
    """
    c = max_num
    if capsules:
        for capsule in capsules:
            val = manhattanDistance(position, capsule)
            c = min(c, val)
    evaluation += 1.0 / c * 0.7

    return evaluation

# Abbreviation
better = betterEvaluationFunction