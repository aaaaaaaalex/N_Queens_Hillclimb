
import numpy as np
import time
import math

# chessboard size
N_QUEENS = 8

# for hillclimb
HILL_CLIMB_MAX_EXECUTION_TIME_SECS = 60

# for random restart hillclimb
MAX_ITERATIONS = 1000

# for simulated annealing
ANNEALING_TEMP = 1


def _horizontalCollisions_ (chessboard, currentCol, ignoreDuplicates=True):
    collisions = 0

    # checks for collisions horizontally
    checkRow = chessboard[currentCol]
    for columnIndex in range(currentCol+1, len(chessboard) ):
        if chessboard[columnIndex] == checkRow:
            collisions += 1

    return collisions



def _diagonalCollisions_ (chessboard, currentCol, ignoreDuplicates=True):
    collisions = 0
    # determines whether queens should only 'look right' for collisions (to get an aggregate number of collisions)
    low = 0 if ignoreDuplicates is False else currentCol+1

    checkRow = chessboard[currentCol]
    for columnIndex in range(low, len(chessboard)):
        calcRow1 = checkRow + (currentCol-columnIndex)
        calcRow2 = checkRow - (currentCol-columnIndex)

        if (calcRow1 >= 0 and calcRow1 < len(chessboard)):
            if (chessboard[columnIndex] == calcRow1):
                collisions += 1

        if (calcRow2 >= 0 and calcRow2 < len(chessboard)):
            if (chessboard[columnIndex] == calcRow2):
                collisions += 1        

    return collisions

 
# calculate the total error of the chessboard state
def calculateError (chessboard):
    collisions = np.zeros( len(chessboard), dtype=int )

    for columnIndex in range( len(chessboard) ):
        collisions[columnIndex] += _horizontalCollisions_(chessboard, columnIndex)
        collisions[columnIndex] += _diagonalCollisions_  (chessboard, columnIndex)

    return np.sum(collisions)


# get the chessboard index of the queen with the most collisions
def getQueenMostCollisions (chessboard):
    collisions = np.zeros( len(chessboard), dtype=int)

    for columnIndex in range( len(chessboard)):
        collisions[columnIndex] += _horizontalCollisions_(chessboard, columnIndex, False)
        collisions[columnIndex] += _diagonalCollisions_  (chessboard, columnIndex, False)

    return np.argmax(collisions)


# takes a chessboard state and returns the best neighbour state
def getBestNeighbourState(chessboard, queenIndex):
    bestChessboard = None
    bestError = 999999999
    for row in range( len(chessboard) ):
        # skip the current state
        if row != chessboard[queenIndex]:
            newChessboard = np.copy(chessboard)
            newChessboard[queenIndex] = row
            newError = calculateError(newChessboard)
            if newError < bestError:
                bestError = newError
                bestChessboard = newChessboard

    return bestChessboard, bestError



# returns a chessboard with it's error based on the annealing method
def getNextNeighbourState(chessboard, queenIndex, temperature):
    currentError = calculateError(chessboard)
    chessboardStates = np.zeros( [len(chessboard), ], dtype=dict)
    nextChessboard = None
    nextError = 999999999

    totalProbs = 0
    # construct an array of every neighbour state
    for row in range( len(chessboard) ):
        if row != chessboard[queenIndex]:
            newChessboard = np.copy(chessboard)
            newChessboard[queenIndex] = row
            newError = calculateError(newChessboard)
            deltaE = newError - currentError

            prob = math.exp( deltaE / temperature )
            totalProbs += prob

            chessboardStates[row] = {
                'chessboard' : newChessboard,
                'error'      : newError,
                'deltaE'     : deltaE,
                'prob'       : prob
            }
        else:
            chessboardStates[row] = None



    #drop current state to avoid effectively not moving
    chessboardStates = chessboardStates[ chessboardStates != np.array(None) ]

    # check for better states and return the best if one (or more) exists
    for chessboardState in chessboardStates:
        if chessboardState['deltaE'] < 0 and chessboardState['error'] < nextError:
            nextError = chessboardState['error']
            nextChessboard = chessboardState['chessboard']


    #  if a suitable chessboard is still not found
    if nextChessboard is None:
        
        # construct a probability distribution
        prob_distrib = np.array(0)
        tokens = np.array(0)
        for i in range(len(chessboardStates)):
            if chessboardStates[i]['prob'] is 1: chessboardStates[i] = None
            else:
                prob_distrib = np.append( prob_distrib, ((chessboardStates[i])['prob'])/totalProbs )
                tokens = np.append( tokens, i)

        chessboardStates = chessboardStates[ chessboardStates != np.array(None) ]

        # choose the next state given the probability distribution
        nextState      = np.random.choice(tokens, 1, p=prob_distrib)
        nextChessboard = (chessboardStates[nextState[0]])['chessboard']
        nextError      = (chessboardStates[nextState[0]])['error']




    return nextChessboard, nextError



# simplest hill-climbing algorithm, hillclimb until error is 0 or max-execution-time is reached (or until no better move is found)
def hillClimb():
    global N_QUEENS
    global HILL_CLIMB_MAX_EXECUTION_TIME_SECS

    chessboard = chessboard = np.random.randint(low=0, high=N_QUEENS, size=N_QUEENS)
    error = calculateError(chessboard)

    totalIterations = 0
    startTime = time.time()

    # keep restarting until error of 0 is found or max execution time is exceeded
    while error != 0 and (time.time() - startTime < HILL_CLIMB_MAX_EXECUTION_TIME_SECS):
        totalIterations += 1

        worstQueen = getQueenMostCollisions(chessboard)
        bestChessboard, bestChessboardError = getBestNeighbourState(chessboard, worstQueen)

        # determine if a local minimum has been reached (trigger a restart)
        if bestChessboardError > error: break
        else:
            chessboard = bestChessboard
            error = bestChessboardError
        
        totalIterations += 1
        if error == 0: break # a slight optimization, preemptively check if the new error is 0 before re-iterating

    # return important information regarding the solution
    return {
        'chessboard': chessboard,
        'error': error,
        'iterations': totalIterations }




def randomRestartHillclimb():
    global N_QUEENS
    global MAX_ITERATIONS

    chessboard = None
    error = -1
    totalIterations = 0
    numRestarts = -1

    # keep restarting until error of 0 is found
    while error != 0:
        numRestarts += 1
        chessboard = np.random.randint(low=0, high=N_QUEENS, size=N_QUEENS)
        error = calculateError(chessboard)
        if error == 0: break

        # iterate over the problem according to the max number of iterations, retart if the max is reached
        for i in range(0, MAX_ITERATIONS):
            worstQueen = getQueenMostCollisions(chessboard)
            bestChessboard, bestChessboardError = getBestNeighbourState(chessboard, worstQueen)

            # determine if a local minimum has been reached (trigger a restart)
            if bestChessboardError > error: break
            else:
                chessboard = bestChessboard
                error = bestChessboardError
            
            totalIterations += 1
            if error == 0: break # a slight optimization, preemptively check if the new error is 0 before re-iterating
            

    # return important information regarding the solution
    return {
        'chessboard': chessboard,
        'error': error,
        'iterations': totalIterations,
        'numRestarts': numRestarts }




def simulatedAnnealing():
    global N_QUEENS
    global ANNEALING_TEMP

    chessboard = np.random.randint(low=0, high=N_QUEENS, size=N_QUEENS)
    error = calculateError(chessboard)
    totalIterations = 0
    numRestarts = -1

    # keep iterating until error of 0 is found
    while error != 0:
        numRestarts += 1
        chessboard = np.random.randint(low=0, high=N_QUEENS, size=N_QUEENS)
        error = calculateError(chessboard)

        for i in range(0, MAX_ITERATIONS):
            worstQueen = getQueenMostCollisions(chessboard)

            nextChessboard, nextChessboardError = getNextNeighbourState(chessboard, worstQueen, ANNEALING_TEMP)
            chessboard = nextChessboard
            error = nextChessboardError

            totalIterations += 1

            if error == 0: break
            
    # return important information regarding the solution
    return {
        'chessboard': chessboard,
        'error': error,
        'iterations': totalIterations ,
        'numRestarts': numRestarts}




def main () :
    print("\nStarting Hillclimb...", end="")
    HCtrainingInfo = hillClimb()
    print("Done.", end="")

    print("\nStarting Random Restart Hillclimb...", end="")
    RRtrainingInfo = randomRestartHillclimb()
    print("Done.", end="")

    print("\nStarting Simulated Annealing...", end="")
    SAtrainingInfo = simulatedAnnealing()
    print("Done.")


    print("\nRR chessboard: {}".format(HCtrainingInfo['chessboard'] ))
    print("RR error: {}".format(HCtrainingInfo['error'] ))
    print("RR iterations: {}".format( HCtrainingInfo['iterations'] ))

    print("\nRR chessboard: {}".format(RRtrainingInfo['chessboard'] ))
    print("RR error: {}".format(RRtrainingInfo['error'] ))
    print("RR iterations: {}".format( RRtrainingInfo['iterations'] ))
    print("RR restarts: {}".format( RRtrainingInfo['numRestarts'] ))

    print("\nSA chessboard: {}".format(SAtrainingInfo['chessboard'] ))
    print("SA error: {}".format(SAtrainingInfo['error'] ))
    print("SA iterations: {}".format( SAtrainingInfo['iterations'] ))
    print("SA restarts: {}".format( SAtrainingInfo['numRestarts'] ))


main()