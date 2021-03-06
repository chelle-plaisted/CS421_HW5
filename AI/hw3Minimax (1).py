import sys

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

##
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
    BREADTH_LIMIT = 8

    def __init__(self, inputPlayerId):
        super(AIPlayer, self).__init__(inputPlayerId, "Minimax")

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
        me = currentState.whoseTurn
        return self.recursiveMoveFinder(currentState, 0, me, -2, 2)


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
            nodes = [Node(move, getNextStateAdversarial(state, move),
                          self.performanceMeasure(getNextStateAdversarial(state, move), me, state.whoseTurn)) for move
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
    # This agent doesn't learn
    #
    def registerWin(self, hasWon):
        # method template, not implemented
        pass

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
        # startingCoord = move.coordList[0]
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
