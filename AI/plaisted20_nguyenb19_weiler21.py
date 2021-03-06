import sys
import random
import math
import time
sys.path.append("..")  # so other modules can be found in parent dir
from Player import *
from AIPlayerUtils import *
from GameState import *

##
# Node
# Description: Represents a node in a search tree that contains the move a player would take,
# the state that would result from taking that move, and the evaluation of that state based on
# the performance measure.
class Node:
    def __init__(self, move=None, state=None, utility=None):
        self.move = move
        self.state = state
        self.eval = utility
    def __str__(self):
        return str(self.eval)

    def __repr__(self):
        return str(self)

##comment
# AIPlayer
# Description: The responsibility of this class is to interact with the game by
# deciding a valid move based on a given game state. This class has methods that
# will be implemented by students in Dr. Nuxoll's AI course.
#
# Variables:
#   playerId - The id of the player.
#   DEPTH_LIMIT - the maximum depth the search tree will expand to
#   BREADTH_LIMIT - the maximum number of child nodes the tree will expand at any depth
##
class AIPlayer(Player):
    DEPTH_LIMIT = 3
    BREADTH_LIMIT = 5

    def __init__(self, inputPlayerId):
        super(AIPlayer, self).__init__(inputPlayerId, "Michael Scott")
        # neural network instance variables
        self.inputs = [0] * 23 # will be length 343 with commented out code
        self.nodeList = [] # list of (state, score) for learning
        self.learningWeight = 0.5 # TODO : test and edit if needed
        self.numHiddenNodes = 16 # set to 2/3 * 1 + len(self.inputs)
        self.training = False # TODO set to false when not training
        self.weights = self.initializeWeights()
        self.me = None # id of this player
        self.outputs = [] # outputs of the hidden layer

    ##
    # getPlacement
    #
    # Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    # Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    # Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        numToPlace = 0
        # implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:  # stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    # Choose any x location
                    x = random.randint(0, 9)
                    # Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    # Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        # Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:  # stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    # Choose any x location
                    x = random.randint(0, 9)
                    # Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    # Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        # Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]
    ##
    # getMove
    # Description: Gets the next move from the Player.
    #
    # Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    # Return: The Move to be made
    ##
    def getMove(self, currentState):
        self.me = currentState.whoseTurn
        return self.recursiveMoveFinder(currentState, 0, self.me, -2, 2)

    ##
    # recursiveMoveFinder
    #
    # Description: recursively expands nodes up to a depth limit to find a desirable move
    #
    # Parameters:
    #   state - the state associated with the node being expanded
    #   depth - the current depth of the recursion
    #   me - my player id
    #   parentMin - the min value of the parent node
    #   parentMax - the max value of the parent node
    #
    # Return: There are three possible returns
    #   Case: Below depth limit and this is the init recursive call. Return the move to make.
    #   Case: Below depth limit; not init call; my turn. Return max evaluation of child nodes.
    #   Case: Below depth limit; not init call; enemy turn. Return min evaluation of child nodes.
    #   Case: Below depth limit; not init call; node should be pruned. Return invalid value.
    #   Case: At the depth limit. Return the evaluation to be used for calculating average score.
    ##
    def recursiveMoveFinder(self, state, depth, me, parentMin, parentMax):
        evaluation = 0
        # initialize range to include all possible values for an evaluation
        currMin = -2
        currMax = 2
        INVALID_EVAL = -10
        if depth < self.DEPTH_LIMIT:
            moves = listAllLegalMoves(state)
            if self.training:
                nodes = [Node(move, getNextStateAdversarial(state, move),
                          self.performanceMeasure(getNextStateAdversarial(state, move), me, state.whoseTurn)) for move
                     in moves]
                self.nodeList += nodes
            else:
                nodes = [Node(move, getNextStateAdversarial(state, move),
                          self.neuralNetwork(getNextStateAdversarial(state, move))) for move
                     in moves]
            # prune all but the best BREADTH_LIMIT nodes
            nodes = self.initialPrune(nodes)
            # sort nodes from lowest to highest if min node in order to more effectively prune
            if state.whoseTurn == 1-me:
                nodes = sorted(nodes, key=self.getEval)

            for node in nodes:
                node.eval = self.recursiveMoveFinder(node.state, depth + 1, me, currMin, currMax)
                # post-recursion, continue to next node if child node evaluates as invalid
                if node.eval == INVALID_EVAL:
                    continue
                # update min value of range if max node
                if state.whoseTurn == me and node.eval > currMin:
                    currMin = node.eval
                # update max value of range if min node
                elif state.whoseTurn != me and node.eval < currMax:
                    currMax = node.eval
                # Alpha-Beta pruning: do not evaluate remaining child nodes if current node is determined invalid
                if currMin >= parentMax or currMax <= parentMin:
                    return INVALID_EVAL
            # prune nodes whose values are invalid
            nodes = [node for node in nodes if node.eval != INVALID_EVAL]
            evaluations = [node.eval for node in nodes]
            # select max child node if current node is a max node
            if state.whoseTurn == me:
                evaluation = max(evaluations)
            # select min child node if current node is a min node
            elif state.whoseTurn != me:
                evaluation = min(evaluations)

        elif depth > 0:
            return self.performanceMeasure(state, me, state.whoseTurn)

        if depth == 0:
            best_node = self.findBestChild(nodes)
            return best_node.move

        else:
            return evaluation

    ##
    # performanceMeasure
    #
    # Description: Assigns a value on the interval [-1.0,1.0] to a state, for use in evaluating the
    #              nodes corresponding to each state.
    #
    # Parameters:
    #   state - the state (potential or current) of the game to be evaluated
    #   me - my player ID
    #   turn - the turn ID for the given state
    #
    # Return: The evaluation of the state
    ##
    def performanceMeasure(self, state, me, turn):
        #evaluate state from my point of view
        state.whoseTurn = me
        myWorkers = getAntList(state, me, (WORKER,))

        if getWinner(state) == 1:
            return 1.0
        elif getWinner(state) == 0:
            return -1.0

        # sum of evaluations of the ants this agent might control, and ants of the enemy
        x = self.workerPositioning(state, myWorkers) \
            + self.queenPositioning(state) \
            + self.soldierPositioning(state) \
            + self.enemyPositioning(state, me)
        # set the state turn back to its original value
        state.whoseTurn = turn
        # the return expression is bounded asymptotically, and will always be on the interval (-1,1)
        return -2 / (1 + 1.05 ** x) + 1

    ##
    # enemyPositioning
    #
    # Description: Evaluates the position of the queen in a given game state
    #
    # Parameters:
    #   state - the state to be evaluated
    #   me - my player ID
    #
    # Return: Integer score of how desirable the state is, per the Queen's location.
    ##
    def enemyPositioning(self, state, me):
        enemyHill = getConstrList(state, 1-me, (ANTHILL,))[0]
        enemyTunnel = getConstrList(state, 1-me, (TUNNEL,))[0]
        enemyArmy = getAntList(state, 1-me, (DRONE,SOLDIER,R_SOLDIER,))
        enemyWorkers = getAntList(state, 1-me, (WORKER,))
        enemyQueen = getAntList(state, 1-me, (QUEEN,))[0]
        myInv = getCurrPlayerInventory(state)
        myQueen = myInv.getQueen()

        x = 0
        # the less ants the enemy has, the better
        if len(enemyWorkers) == 0:
            x += 40
        x += (-2 * len(enemyArmy))
        x += (50 - (5 * enemyQueen.health))

        for worker in enemyWorkers:
            # 12 because basically everywhere is reachable in 12 steps, and we want closer positions to be a larger
            # negative increment to the score per worker than distant ones
            if worker.carrying:
                x -= (12 - min(approxDist(worker.coords, enemyHill.coords),
                               approxDist(worker.coords, enemyTunnel.coords)))

        # the closer the enemy's attacking ants are to my queen, the worse
        for enemy in enemyArmy:
            dist = approxDist(enemy.coords, myQueen.coords)
            x -= (25 - dist)

        return x

    ##
    # workerPositioning
    #
    # Description: Evaluates the position of each worker in a given game state
    #
    # Parameters:
    #   state - the state to be evaluated
    #   myWorkers - list of Ant() objects corresponding to the workers this agent controls
    #
    # Return: Integer score of how desirable the state is, per the workers' locations.
    ##
    def workerPositioning(self, state, myWorkers):
        if len(myWorkers) > 2 or len(myWorkers) == 0:
            return -90  # arbitrarily bad, always want 2 workers
        elif len(myWorkers) == 1:
            return 0

        myInv = getCurrPlayerInventory(state)
        myTunnel = myInv.getTunnels()[0]
        myHill = myInv.getAnthill()
        myFood = getCurrPlayerFood(self, state)

        x = 0
        for worker in myWorkers:
            # 12 because basically everywhere is reachable in 12 steps, and we want closer positions to be a larger
            # positive increment to the score per worker than distant ones
            if worker.carrying:
                x += (12 - min(approxDist(worker.coords, myHill.coords),
                               approxDist(worker.coords, myTunnel.coords)))
                # Encourage worker to go on hill/tunnel if it has food
                if min(approxDist(worker.coords, myHill.coords), approxDist(worker.coords, myTunnel.coords)) == 0:
                    x += 20
            else:
                x += (12 - min(approxDist(worker.coords, myFood[0].coords),
                               approxDist(worker.coords, myFood[1].coords)))
                # Encourage worker to go on food if it does not have food
                if min(approxDist(worker.coords, myFood[0].coords), approxDist(worker.coords, myFood[1].coords)) == 0:
                    x += 20
        return x

    ##
    # queenPositioning
    #
    # Description: Evaluates the position of the queen in a given game state
    #
    # Parameters:
    #   state - the state to be evaluated
    #
    # Return: Integer score of how desirable the state is, per the Queen's location.
    ##
    def queenPositioning(self, state):
        x = 0
        myFoods = getCurrPlayerFood(self, state)
        myInv = getCurrPlayerInventory(state)
        myQueen = myInv.getQueen()
        myHill = myInv.getAnthill()
        myTunnel = myInv.getTunnels()[0]
        # The queen should avoid standing on food, the anthill or the tunnel
        for food in myFoods:
            if myQueen.coords == food.coords:
                return -90
        if approxDist(myQueen.coords, myHill.coords) == 0 or approxDist(myQueen.coords, myTunnel.coords) == 0:
            return -90
        # So long as she is blocking other ants, the queen should stay close to the anthill, less health is worse
        else:
            x += (-2 * myQueen.health)
            x += (6 - approxDist(myQueen.coords, myHill.coords))
            return x

    ##
    # soldierPositioning
    #
    # Description: Evaluates the position of this agent's soldiers in a given game state
    #
    # Parameters:
    #   state - the state to be evaluated
    #
    # Return: Integer score of how desirable the state is, per the soldiers' locations.
    ##
    def soldierPositioning(self, state):
        me = state.whoseTurn
        myAnts = getAntList(state, me)
        enemyWorkers = getAntList(state, 1-me, (WORKER,))

        # bugfix: the agent would't prefer building soldiers to drones or r_soldiers
        for ant in myAnts:
            if ant.type == R_SOLDIER or ant.type == DRONE:
                return -500

        # target enemy workers first, then attack queen
        if len(enemyWorkers) > 0:
            target = enemyWorkers[0].coords
        else:
            target = getEnemyInv(None, state).getQueen().coords

        x = 0
        soldierList = getAntList(state, me, (SOLDIER,))
        # discourage building more than two soldiers
        if len(soldierList) > 2:
            x -= 200

        # similar to workers, lesser distance to target is preferable to equal or greater
        for soldier in soldierList:
            dist = approxDist(soldier.coords, target)
            x += (25 - dist)

        return x

    ##
    # foodClosestTo
    #
    # Description: finds the food closest to a source
    #
    # Parameters:
    #   state - the state to be evaluated
    #   source - the source to find food near to.
    #
    # Return: coords of the food closest to source.
    ##
    def foodClosestTo(self, state, source):
        allFoods = getCurrPlayerFood(self, state)
        bestDistSoFar = 1000  # i.e., infinity
        bestFoodCoords = ()
        for food in allFoods:
            dist = approxDist(source.coords, food.coords)
            if dist < bestDistSoFar:
                bestDistSoFar = dist
                bestFoodCoords = food.coords
        return bestFoodCoords


    ##
    # findBestChild
    #
    # Description: given a list of nodes, return the node with the highest score
    #
    # Parameters:
    #   nodes - the list of nodes to maximize
    #
    # Return: the best node
    ##
    def findBestChild(self, nodes):
        max = -2
        bestChild = None
        for node in nodes:
            val = node.eval
            if val >= max:
                bestChild = node
                max = val
        return bestChild

    ##
    # getEval
    #
    # Description: get the assigned score of a node
    #
    # Parameters:
    #   node - the node in question
    #
    # Return: the node's score
    ##
    def getEval(self, node):
        return node.eval

    ##
    # prune
    #
    # Description: prune nodes the agent probably doesn't need to consider
    #
    # Parameters:
    #   nodes - the list of nodes to prune
    #
    # Return: sorted list of the best BREADTH_LIMIT nodes, ranked by getEval
    ##
    def initialPrune(self, nodes):
        return sorted(nodes, key=self.getEval, reverse=True)[:self.BREADTH_LIMIT]

    ##
    # getAttack
    # Description: Gets the attack to be made from the Player
    #
    # Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        # Don't care
        return enemyLocations[0]

    ##
    # registerWin
    #
    # Run the neural network at the end of the tournament during learning
    #
    def registerWin(self, hasWon):
		# call self.neuralNetwork(state, True, score) for every state score pair in
        if self.training:
            random.shuffle(self.nodeList)
            for element in self.nodeList:
                if element.state.whoseTurn == self.me:
                    self.neuralNetwork(element.state, element.eval)
            # print the weights
            print('Final weights: ', self.weights)
		# reset the state-score map
        self.nodeList = []

    ############################### NEURAL NETWORK FUNCTIONS ####################

    ##
    # initializeWeights
    #
    # Description: initialize the weights to a list of proper length. Set starting values
    # [-1, 1]
    #
    # Return: a list of hardcoded weights
    ##
    def initializeWeights(self):
        # need weights for every input (which includes bias) * number of hidden nodes
        # need weights for output * number of hidden nodes + 1 for output bias
        length = len(self.inputs) * self.numHiddenNodes + 1 * self.numHiddenNodes + 1
        rtn = []
        if self.training:
            for i in range(0,length): # randomize weights
                rtn += [random.uniform(-1, 1)]
        else:
            # hard coded list for when not training
            rtn =  [-28.691697088560502, -33.46609076384445, -42.27798376936759, -0.40466454499923976, -0.3473845885928241, -43.96378572614014, 28.864244153725565, -57.323754891844025, -58.161905430847874, -1.8068618367886478, -31.315964852789765, 9.199340203671065, 10.201348387009086, 11.403106743870795, 30.59915192485749, -0.015221268483609121, 8.002095539859079, -10.58927868269759, -2.274353968482503, -20.640928499512984, -2.8645835481330266, 3.1944586979682246, -31.3754797648029, -29.806594441683416, -21.881699785267166, -48.15956921884178, -14.957574795609522, -0.49171501270965606, 6.558641816031159, 38.02994177127504, -56.07615476550792, -57.718959842938666, -25.59089405962987, 6.805452191798216, 15.235463167651101, 9.822050330208272, 25.213920008400116, 31.047287754949714, 6.8879603321395635, 11.473829715842848, -4.788021844974098, -3.6465700353506767, -5.1565387147865955, -2.6674517532564903, 7.6909385795827525, -4.299183290120432, 31.61136941060744, 26.599243987188895, 33.10221741396366, -2.980827115410982, -15.66320379431509, -0.988546659017363, -38.71988241908972, 56.05859876066779, 56.97485145560082, -21.8473527425553, -1.4016973945217128, -17.573288624656367, -8.290352953337152, -27.61138177646217, -22.28951771919116, -13.278830282183888, -12.05543431284663, 1.54449864333425, 6.3809974096510675, 8.898456726457697, 1.4321837341131578, -7.813128264208954, 25.727505513497267, 27.414495923602757, 22.113976123182415, 31.98302337647038, -3.5961538951763323, -12.211151496044062, -1.7536646676688274, -29.731144893060765, 50.24036716070851, 50.99419905608709, 4.238344521532708, -0.5831193284205003, -14.655404482056449, -7.635878163253869, -24.998938185297465, -21.354655025830393, -13.834601620362283, -10.065602404209784, 4.030648843053688, 4.297715467373672, 12.338577069737863, 5.3461410791652915, -3.1601304675700157, 14.611198207238225, 45.453738306339545, 39.011110126891495, 48.62603476496611, -9.78434859712188, -23.383540386154817, 0.7967447159274396, -44.19203784685934, 82.89960875706582, 80.85283339515277, 0.745793862283302, 14.825419424856134, -27.08657848516362, -14.234858107248286, -37.882344799838, -33.64738227639466, -13.804558900046096, -9.408770306852949, 7.678231893004934, 5.097983256470402, 19.549553456775236, 7.648475142796521, -12.106671348013796, 21.692174539795392, 0.5597048562270195, 2.6070123306605697, 1.497004341382454, 1.4622354074338195, -2.7223884686054802, 0.012957922392238756, 1.4610339362574338, 2.580838799613958, -4.339446590380908, -0.856003187166365, 4.438109165332269, -5.527253864195456, 2.5679501660541137, -1.1805058658288394, 0.30688594910567474, -2.977623501216087, 4.325925559054465, -1.9228388672871224, -1.1264328163858077, -1.9130039085790882, 0.4958498697718199, -1.5500244006866941, -27.472137085663743, -26.497509253881923, -22.36147687382072, -33.200553119907816, 1.7254055685306025, 9.546317256066839, 11.27487790268445, 29.53308921157677, -49.82138482954769, -53.623913135243676, 23.089914546833096, -17.150952101273248, 13.802093386586401, 8.856249139905731, 24.966142092096852, 23.170773979056513, 12.86319390882139, 10.166910428112866, -2.523081069082389, -5.428604434644269, -12.824530273351764, -8.465644568125965, 13.131625372641675, -11.678052289931625, -1.3240946292762912, -0.8366040333712268, 0.5595355547401584, 5.041306332469354, 0.6701624772178508, 0.2956793789150064, -1.3608633271261703, 3.082223027694365, -4.627332776634112, 2.283925201949049, 0.5993701332683102, 3.6313764698636555, 0.1696540440262362, 3.3110349529879195, -5.6972408280925695, 4.616188017976032, -3.6460387241378376, 2.4592586593359957, 3.4893980120908057, 7.457138850900083, 35.88538560113585, -11.596898678875604, 0.2064596586176959, 32.77695520658485, 27.654736506274826, 50.23897093540923, -2.3458059986653215, -8.536429269111832, -8.938014861571013, -34.4670258498223, 57.746179871085126, 62.98930646337554, -2.830065167897824, -5.823123280749632, -19.334538445593587, -9.484581507241808, -26.980336252505452, -26.53026353412114, -13.040609105141325, -12.001991288616416, 3.721605591227008, 5.34714933068324, 18.705157497355614, 5.798364294648476, -7.303743773868916, 4.990226737855663, -0.12719273087132854, 5.078321543543674, -4.196964715786052, -0.6123038837106499, 0.2356064305518399, 5.1764956354435725, -8.955346789039503, 4.08796027216774, -0.7635726361644821, -4.7762676751373725, -1.7400850987551821, 3.218703754352026, -2.30960159514159, 2.3778872010642287, -0.04526099103191726, 5.096872731575702, -1.5440461446857787, 2.9541877729827144, 2.3697185241376504, 5.020978646854596, -0.5553904963671654, 1.154491263766373, 5.670673944382372, 51.15361906622684, 43.774700475482106, 56.71340283936031, -7.74486187941854, -25.96505034237653, -0.9399191394925899, -54.3646238113104, 91.26994776345205, 91.34638153515571, 0.0521979635155133, 0.40940630573676257, -29.027506485112514, -13.366086411738621, -32.846122910376714, -50.192888879993646, -18.894366151805084, -19.78977983958741, 7.042161882869439, 6.671257111984603, 15.483383162524573, 6.088459192921776, -14.277163613413677, 26.424195162196646, 24.685311268922355, 25.860399435056475, 24.41646043125329, 6.944524731051245, -2.85062238859471, 4.709159018740957, -16.799445990185745, 46.39482875819729, 47.6406041642375, -0.3075051875975554, 2.2084498143548266, -10.147529014619227, -22.374125018255118, -9.648248211677894, -23.014818071084402, -2.2574370663280594, -15.452823396921854, 12.352434155806117, 0.7116175123407668, 35.160976045107034, -0.7797413620239791, 8.957080689224593, 39.41595026607982, 33.14451664738501, 28.0692554908866, 43.07116084795297, -3.421355159911554, -11.571200874743635, 5.490713689929006, -36.77743688863649, 63.07130431844955, 64.00044970476478, 13.804759138235273, 1.466270414977838, -18.857761167233317, -10.799870890755773, -30.294318937319492, -26.496274863829154, -15.036667596135622, -10.212044608203644, 5.325549314582047, 5.060921349674873, 14.1924009329761, 7.971935496492935, -10.27751662678912, 15.064955422624699, -15.824470237288562, -15.627954055168363, -21.300304371799, 2.957212937436831, 8.701470442917964, -0.1833168365084802, 17.13656969243121, -32.68044675008497, -34.28582123374593, 0.21690864623073824, -0.7657686327000462, 10.041561622251484, 7.028923704826085, 12.310205788776363, 13.653373902923095, 4.320090660314275, 6.920336040371736, -0.7279147089000748, -1.5916573623927681, -13.410502730188362, 3.363486778319755, 5.422223610095854, -3.957747921379577, -47.593930957221595, -49.32780447560053, -54.58704728162975, 12.950219607433777, 8.516536691080129, 5.999374594503474, 42.942701088093315, -108.98667484426011, -104.71688934901654, 0.17677134450197743, 0.7222442811085157, 21.586744317906057, 5.237407705593972, 7.292380665951421, 59.26259672753116, -6.745339806269106, 7.991451442629405, -16.488966300766872, -13.138493411908803, -40.475087464185634, -9.732721892457565, 9.32581439424313, -62.95627277631282, -11.346271664309851, -27.269298796954914, -34.691067780756875, -34.530986082571324, -15.943867763951411, 0.13743103210133353, 56.781675879397135, 2.484701907626868] # TODO set to hardcoded list
        return rtn

    ##
    # getLocationInputs
    #
    # Description: map the cells the of the playing area to a list, with
    # false values if there is no ant there, and true if there is
    #
    # Parameters:
    #   antList : list of ants to check
    #   length : length of list
    #   bottom : boolean for if the ants are on the bottom
    #   fullRange : does the ant have full range of the board
    # Return : the cells
    ##
    def getLocationInputs(self, antList, length, bottom, fullRange = True):
        newCells = [-1] * length
        if antList is None:
            return newCells
        # reset the cells where ants are located to be True
        for ant in antList:
            if ant is None:
                continue
            x = ant.coords[0]
            y = ant.coords[1]
            if not fullRange and (bottom and y < 6) or (not bottom and y > 3):
                # not in range
                continue

            if bottom:
                # on bottom half
                x = 9 - ant.coords[0]
                y = 9 - ant.coords[1]
            newCells[x + y * 10] = True
        return newCells

    ##
    # mapInputs
    #
    # Description: map the relevant information from the state to an input array containing
    # values in the range [-1, 1]. Store input array in self.inputs. Includes a bias value of 1.
    #
    # Parameters:
    #   state: the state to generate inputs for
    #   me: Scottie's id
    ##
    def mapInputs(self, state):
        # general info
        turn = state.whoseTurn
        state.whoseTurn = self.me
        enemyHill = getConstrList(state, 1-self.me, (ANTHILL,))[0]
        enemyTunnel = getConstrList(state, 1-self.me, (TUNNEL,))[0]
        enemyArmy = getAntList(state, 1-self.me, (DRONE,SOLDIER,R_SOLDIER,))
        enemyWorkers = getAntList(state, 1-self.me, (WORKER,))
        enemyQueen = getAntList(state, 1-self.me, (QUEEN,))
        if len(enemyQueen) > 0:
            enemyQueen = enemyQueen[0]
        else :
            enemyQueen = None
        myHill = getConstrList(state, self.me, (ANTHILL,))[0]
        myTunnel = getConstrList(state, self.me, (TUNNEL,))[0]
        myInv = getCurrPlayerInventory(state)
        myQueen = myInv.getQueen()
        myWorkers = getAntList(state, self.me, (WORKER,))
        mySoldiers = getAntList(state, self.me, (SOLDIER,))
        myRSoldiers = getAntList(state, self.me, (R_SOLDIER,))
        myDrone = getAntList(state, self.me, (DRONE,))
        myFoods = getCurrPlayerFood(self, state)
        state.whoseTurn = turn
        inputs = []

        ### LOCATION ###
        # tell if angent is on bottom for location processing
        if myHill.coords[1] > 3:
            onBottom = True
        else:
            onBottom = False

        # Scottie's ants
        # inputs += self.getLocationInputs(myWorkers, 40, onBottom, False) # workers
        # inputs += self.getLocationInputs([myQueen], 40, onBottom, False) # queen
        # inputs += self.getLocationInputs(mySoldiers, 100, onBottom, True) # soldiers
        # enemy ants
        # inputs += self.getLocationInputs(enemyWorkers, 40, onBottom, False) # workers
        # inputs += self.getLocationInputs(enemyArmy, 100, onBottom, True) # army

        # rest of queen
        if not myQueen is None and (approxDist(myQueen.coords, myHill.coords) == 0 or approxDist(myQueen.coords, myTunnel.coords) == 0):
            inputs.append(-0.5)
        else:
            inputs.append(0.5)
        count = 0
        if not myQueen is None:
            for food in myFoods:
                if myQueen.coords == food.coords:
                    count += 1
        inputs.append(count * 0.25)

        ### NUMBER OF ANTS ###

        if len(myWorkers) == 0 or len(myWorkers) > 2: # my workers
            inputs.append(True)
        else:
            inputs.append(-1)
        if len(myWorkers) == 1:
            inputs.append(True)
        else:
            inputs.append(-1)
        if len(myWorkers) == 2:
            inputs.append(True)
        else:
            inputs.append(-1)

        if len(mySoldiers) > 2: # my soldiers
            inputs.append(True)
        else:
            inputs.append(-1)
        if len(mySoldiers) == 0:
            inputs.append(0)
        elif len(mySoldiers) == 1:
            inputs.append(0.2)
        elif len(mySoldiers) == 2:
            inputs.append(0.4)

        if len(myRSoldiers) > 0: # my ranged soldiers
            inputs.append(True)
        else:
            inputs.append(-1)

        if len(myDrone) > 0: # my drones
            inputs.append(True)
        else:
            inputs.append(-1)

        inputs.append(0.02 * len(enemyArmy)) # enemy army
        inputs.append(0.04 * len(enemyWorkers)) # enemy workers

        if len(enemyWorkers) == 0:
            inputs.append(True)
        else:
            inputs.append(-1)

        ### FOOD DISTANCE ###
        #My ants: average distance of all carrying workers to food drop off, average distance of all not carrying workers to food
        #enemy ants:  average distance of all carrying workers to food drop off, average distance of all not carrying workers to food

        # my workers
        numCarry = 0
        sumCarry = 0
        numForage = 0
        sumForage = 0

        for worker in myWorkers:
            # 12 because basically everywhere is reachable in 12 steps, and we want closer positions to be a larger
            # positive increment to the score per worker than distant ones
            if worker.carrying:
                numCarry += 1
                sumCarry += (12 - min(approxDist(worker.coords, myHill.coords),
                               approxDist(worker.coords, myTunnel.coords)))
                # Encourage worker to go on hill/tunnel if it has food
                if min(approxDist(worker.coords, myHill.coords), approxDist(worker.coords, myTunnel.coords)) == 0:
                    sumCarry += 20
            else:
                numForage += 1
                sumForage += (12 - min(approxDist(worker.coords, myFoods[0].coords),
                               approxDist(worker.coords, myFoods[1].coords)))
                # Encourage worker to go on food if it does not have food
                if min(approxDist(worker.coords, myFoods[0].coords), approxDist(worker.coords, myFoods[1].coords)) == 0:
                    sumForage += 20
        if numCarry > 0:
            inputs.append((sumCarry / float(numCarry)) / 32)
        else:
            inputs.append(0)

        if numForage > 0:
            inputs.append((sumForage / float(numForage)) / 32)
        else:
            inputs.append(0)

        # enemy workers
        # the less ants the enemy has, the better
        numCarry = 0
        sumCarry = 0
        for worker in enemyWorkers:
            # 12 because basically everywhere is reachable in 12 steps, and we want closer positions to be a larger
            # negative increment to the score per worker than distant ones
            if worker.carrying:
                numCarry += 1
                sumCarry -= (12 - min(approxDist(worker.coords, enemyHill.coords),
                               approxDist(worker.coords, enemyTunnel.coords)))
        if numCarry > 0:
            inputs.append((sumCarry / float(numCarry)) / 32)
        else:
            inputs.append(0)

        ### SOLDIER DISTANCE ###
        # soldier distance : avg / 25
        # my ants: average distance of all soldiers to their target
        # my soldiers
        # target enemy workers first, then attack queen
        if len(enemyWorkers) > 0:
            target = enemyWorkers[0].coords
        elif not enemyQueen is None:
            target = enemyQueen.coords
        else:
            target = enemyHill.coords

        numArmy = 0
        sumArmy = 0
        # similar to workers, lesser distance to target is preferable to equal or greater
        for soldier in mySoldiers:
            numArmy  += 1
            dist = approxDist(soldier.coords, target)
            sumArmy += (25 - dist)
        if numArmy > 0:
            inputs.append((sumArmy / float(numArmy)) / 25)
        else:
            inputs.append(0)

        # enemy army
        numArmy = 0
        sumArmy = 0
        if myQueen is None:
            target = myHill.coords
        else:
            target = myQueen.coords
        for enemy in enemyArmy:
            numArmy += 1
            dist = approxDist(enemy.coords, target)
            sumArmy -= (25 - dist)

        if numArmy > 0:
            inputs.append((sumArmy / float(numArmy)) / 25)
        else:
            inputs.append(0)

        ### HEALTH ###
        if enemyQueen is None:
            inputs.append(-1)
        elif myQueen is None or enemyQueen.health >= myQueen.health:
            inputs.append(True)
        else:
            inputs.append(-1)

        if not enemyQueen is None:
            inputs.append(-1 + 0.2 * enemyQueen.health)
        if not myQueen is None:
            inputs.append(-1 + 0.2 * myQueen.health)

        ### FOOD AMOUNT ###
        inputs.append(state.inventories[self.me].foodCount / 11)
        inputs.append(state.inventories[1 - self.me].foodCount / 11)

        ### BIAS ###
        inputs.append(1)

        self.inputs = inputs

    ##
    # neuralNetwork
    #
    # Description: complete the steps to run the neural network and learn if relevant
    #
    # Parameters:
    #   state: GameState being evaluated with neural network
    #   me: this AI's player Id
    #   training: boolean value to include if you want the agent to learn through
    #       backpropogation. Defaults to False.
    #   score: score from evaluation function for learning. Only include if training.
    #       Defaults to None.
    #
    # Return : the evaluation score determined by the neural network.
    ##
    def neuralNetwork(self, state, score = None):
        # map inputs correctly
        self.mapInputs(state)

        # run network by running input array through network
            # multiply by weight
            # sum for each node
            # apply activation function: g(x) = 1 / 1- e^x
        output = self.runNetwork()

        if self.training:
            # calculate error (compare)
            error = score - output
            # apply backpropogation to update weights stored in instance variables
            self.backpropogate(error, output)
        return output

    ##
    # runNetwork
    #
    # Description: use instance variables for inputs and weights to run the network
    # and return an output evalutation.
    #
    # Return : the evaluation score determined by the network
    ##
    def runNetwork(self):
        outputs = []
        # run inputs through hidden layer to get outputs
        # each input gets run for each hidden node
        for i in range(0, self.numHiddenNodes):
            # get weighted sum
            total = 0
            counter = i * len(self.inputs) # get to correct set of weights
            subweights = self.weights[counter:counter + len(self.inputs)]
            mult = [a*b for a,b in zip(self.inputs, subweights)]
            total = sum(mult)

            # apply activation function
            try:
                result = 1 / (1 + math.exp(-total))
            except:
                # print('weights: ', self.weights)
                print('inputs: ', self.inputs)
                print('mult: ', mult)
                print('weights: ', self.weights)
                print('error 1: x = ', total)
                sys.exit()
            outputs.append(result)
        # with hidden node values, propogate to output node
        total = 0
        counter = self.numHiddenNodes * len(self.inputs) # starting index in weights
        self.outputs = outputs
        outputs.append(1) # bias
        # weighted total
        subweights = self.weights[counter:counter + len(outputs)]
        mult = [a*b for a,b in zip(outputs, subweights)]
        total = sum(mult)
        # apply activation function
        try:
            result = 1 / (1 + math.exp(-total))
        except:
            print('error 1: x = ', total)
        # print(result)
        return result

    ##
    # backpropogate
    #
    # Description: apply backpropogation algorithm (see assignment description for notes
    # on resources) and update weights stored in instance variables
    #
    # Parameters:
    #   error : error of network
    ##
    def backpropogate(self, error, score):
        # calculate error term (error * g'(in) ) of output nodes
        outputErrorTerm = error * score * (1-score)

        # calculate error term of hidden nodes: error * g'(in)
        errorTerms = []
        # calculate error of hidden nodes: outputTermError * weight b/w hidden node & output
        outputWeights = self.weights[-self.numHiddenNodes:]
        errors = [x * outputErrorTerm for x in outputWeights]
        errorTerms = [a*b*(1-b) for a,b in zip(errors, self.outputs)]
        # adjust weights : use same structure as running network to ensure correct
        # weights adjusted

        # between inputs and hidden nodes
        weights = []
        for i in range(0, self.numHiddenNodes):
            counter = i * len(self.inputs) # get to correct set of inputs
            subweights = self.weights[counter:counter + len(self.inputs)]
            weights += [a + self.learningWeight * errorTerms[i] * b for a,b in zip(subweights, self.inputs)]

        # between hidden nodes and output node
        self.outputs.append(1) # output node bias
        counter = self.numHiddenNodes * len(self.inputs) # starting index in weights connected to output node
        subweights = self.weights[counter:counter + len(self.outputs)]
        weights += [a + self.learningWeight * outputErrorTerm * b for a,b in zip(subweights, self.outputs)]

        self.weights = weights


##
# Revised version of getNextStateAdversarial from AIPlayerUtils
# Correctly Updates the hasMoved attribute for an ant that has moved
#
def getNextStateAdversarial(currentState, move):
    # variables I will need
    nextState = getNextState(currentState, move)
    myInv = getCurrPlayerInventory(nextState)
    myAnts = myInv.ants

    # If an ant is moved update their coordinates and has moved
    if move.moveType == MOVE_ANT:
        startingCoord = move.coordList[len(move.coordList) - 1]
        for ant in myAnts:
            if ant.coords == startingCoord:
                ant.hasMoved = True
    elif move.moveType == END:
        for ant in myAnts:
            ant.hasMoved = False
        nextState.whoseTurn = 1 - currentState.whoseTurn
    return nextState

##
# Revised version of getNextState from AIPlayerUtils
# Does not update the carrying attribute for workers, since that attribute can never
# be changed in a single state in the actual game.
#
def getNextState(currentState, move):
    # variables I will need
    myGameState = currentState.fastclone()
    myInv = getCurrPlayerInventory(myGameState)
    me = myGameState.whoseTurn
    myAnts = myInv.ants
    myAntHill = myInv.getAnthill()

    # If enemy ant is on my anthill or tunnel update capture health
    ant = getAntAt(myGameState, myAntHill.coords)
    if ant is not None:
        if ant.player != me:
            myAntHill.captureHealth -= 1

    # If an ant is built update list of ants
    antTypes = [WORKER, DRONE, SOLDIER, R_SOLDIER]
    if move.moveType == BUILD:
        if move.buildType in antTypes:
            ant = Ant(myInv.getAnthill().coords, move.buildType, me)
            myInv.ants.append(ant)
            # Update food count depending on ant built
            if move.buildType == WORKER:
                myInv.foodCount -= 1
            elif move.buildType == DRONE or move.buildType == R_SOLDIER:
                myInv.foodCount -= 2
            elif move.buildType == SOLDIER:
                myInv.foodCount -= 3
        # ants are no longer allowed to build tunnels, so this is an error
        elif move.buildType == TUNNEL:
            print("Attempted tunnel build in getNextState()")
            return currentState

    # If an ant is moved update their coordinates and has moved
    elif move.moveType == MOVE_ANT:
        newCoord = move.coordList[-1]
        startingCoord = move.coordList[0]
        for ant in myAnts:
            if ant.coords == startingCoord:
                ant.coords = newCoord
                # TODO: should this be set true? Design decision
                ant.hasMoved = False
                # If my ant is close to an enemy ant attack it
                attackable = listAttackable(ant.coords, UNIT_STATS[ant.type][RANGE])
                for coord in attackable:
                    foundAnt = getAntAt(myGameState, coord)
                    if foundAnt is not None:  # If ant is adjacent my ant
                        if foundAnt.player != me:  # if the ant is not me
                            foundAnt.health = foundAnt.health - UNIT_STATS[ant.type][ATTACK]  # attack
                            # If an enemy is attacked and looses all its health remove it from the other players
                            # inventory
                            if foundAnt.health <= 0:
                                myGameState.inventories[1 - me].ants.remove(foundAnt)
                            # If attacked an ant already don't attack any more
                            break
    return myGameState


# Unit tests:
# create new test state
newPlayer = AIPlayer(PLAYER_ONE)
newPlayer.me = 0
testState = GameState.getBasicState()
food1 = Building((5, 0), FOOD, NEUTRAL)
food2 = Building((4, 0), FOOD, NEUTRAL)
testState.board[5][0].constr = food1
testState.board[4][0].contrs = food2
food3 = Building((4, 9), FOOD, NEUTRAL)
food4 = Building((5, 9), FOOD, NEUTRAL)
testState.board[4][9].constr = food3
testState.board[5][9].contrs = food4
testState.inventories[NEUTRAL].constrs += [food1, food2]
testState.inventories[NEUTRAL].constrs += [food3, food4]
p1Worker = Ant((5, 4), WORKER, 0)
testState.board[5][4].ant = p1Worker
testState.inventories[0].ants.append(p1Worker)
testState.inventories[0].foodCount = 10

#test queenPositioning
queenOffHill = getNextState(testState, Move(MOVE_ANT, [(0, 0), (0, 1)], None))
testEval = newPlayer.queenPositioning(testState)
queenEval = newPlayer.queenPositioning(queenOffHill)
if queenEval <= testEval:
    print("Error, queen off hill should be better than on hill")

#test workerPositioning
workerState = getNextState(testState, Move(BUILD, [(0,0)], WORKER))
if newPlayer.workerPositioning(testState, getAntList(testState, 0, (WORKER,))) >= \
        newPlayer.workerPositioning(workerState, getAntList(workerState, 0, (WORKER,))):
    print("Error, 2 workers is better than 1 worker")

#test performanceMeasure
winningState = testState.clone()
winningState.inventories[0].foodCount = 11
losingState = testState.clone()
losingState.inventories[1].foodCount = 11
if not newPlayer.performanceMeasure(winningState, 0, 0) == 1:
    print("Error, player won so rating should be 1")
if not newPlayer.performanceMeasure(losingState, 0, 0) == -1:
    print("Error, player lost so rating should be -1")

# test soldierPositioning
soldierState = getNextState(testState, Move(BUILD, [0,0], SOLDIER))
if newPlayer.soldierPositioning(soldierState) <= newPlayer.soldierPositioning(testState):
    print("Error, one soldier is better than no soldiers")
soldierState2 = getNextState(soldierState, Move(MOVE_ANT, [(0,0), (0,1), (1,1)], None))
if newPlayer.soldierPositioning(soldierState2) <= newPlayer.soldierPositioning(soldierState):
    print("Error, soldier is closer to queen, so it should be better")

# test find best child
nodes = [Node(Move(END, None, None), testState, evaluation) for evaluation in range(0, 30)]
bestNode = newPlayer.findBestChild(nodes)
if not bestNode.eval == 29:
    print("Error in findBestChild(): findBestChild() returns %s instead of 29" % (bestNode.eval))

newPlayer.training = True

# test initializeWeights
newPlayer.inputs = [0] * 5 # 4 inputs + bias
newPlayer.numHiddenNodes = 5
weights = newPlayer.initializeWeights()
if not len(weights) == 31:
    print("Error in initializeWeights(): initializeWeights() makes length %s instead of 31" % (len(weights)))
    print("Inputs: %s; Hidden Nodes: %s" % (len(newPlayer.inputs), newPlayer.numHiddenNodes))

# test runNetwork
newPlayer.inputs = [2, 4, 1] # 1 for bias
newPlayer.weights = [-1, 1, 1, 2, 3, 2, 0, -2, -1, 2, 3, -1, -1]
newPlayer.numHiddenNodes = 3
result = newPlayer.runNetwork()
if not round(result, 3) == 0.980:
    print("Error in runNetwork(): runNetwork() returns %s instead of 0.980 (rounded)" % round(result, 3))

# test backPropogate
newPlayer.inputs = [2, 4, 1] # 1 for bias
newPlayer.weights = [-1, 1, 1, 2, 3, 2, 0, -2, -1, 2, 3, -1, -1]
newPlayer.numHiddenNodes = 3
result = newPlayer.runNetwork()
error = 2 - result
newPlayer.backpropogate(error, result)
newResult = newPlayer.runNetwork()
if not (2 - newResult) < (2 -result):
    print("Error in backpropogate(): backpropogate() alters result from %s to %s with a target result of 2" % (result, newResult))


# bigger test
testState = GameState.getBasicState()
food1 = Building((5, 0), FOOD, NEUTRAL)
food2 = Building((4, 0), FOOD, NEUTRAL)
testState.board[5][0].constr = food1
testState.board[4][0].contrs = food2
food3 = Building((4, 9), FOOD, NEUTRAL)
food4 = Building((5, 9), FOOD, NEUTRAL)
testState.board[4][9].constr = food3
testState.board[5][9].contrs = food4
testState.inventories[NEUTRAL].constrs += [food1, food2]
testState.inventories[NEUTRAL].constrs += [food3, food4]
p1Worker = Ant((5, 4), WORKER, 0)
testState.board[5][4].ant = p1Worker
testState.inventories[0].ants.append(p1Worker)
testState.inventories[0].foodCount = 10

newPlayer = AIPlayer(PLAYER_ONE)
newPlayer.training = True
newPlayer.weights = newPlayer.initializeWeights()
newPlayer.me = 0
desiredScore = newPlayer.performanceMeasure(testState, 0, 0)
old = newPlayer.weights
networkScore = newPlayer.runNetwork()
for i in range(0, 100):
    result = newPlayer.neuralNetwork(testState, desiredScore)
    #newPlayer.mapInputs(testState)
    #result = newPlayer.runNetwork()
    #print(result)
    #error = desiredScore - result
    #newPlayer.backpropogate(error, result)
new = newPlayer.weights
backpropScore = newPlayer.runNetwork()
if not (desiredScore - backpropScore) < (desiredScore - networkScore):
    print("Error in neuralNetwork(): neuralNetwork() alters result from %s to %s with a target result of %s" % (networkScore, backpropScore, desiredScore))
