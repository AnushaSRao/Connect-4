import numpy as np
import time
import random
import math
from keras.models import Sequential,model_from_json
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam



class board():
    def __init__(self):
        self.init_board = np.zeros([6,7]).astype(str)
        self.init_board[self.init_board == "0.0"] = " "
        self.current_board = self.init_board
        
    def drop_piece(self, column, player):
        if self.current_board[0, column] != " ":
            return "Invalid move"
        else:
            row = 0; pos = " "
            while (pos == " "):
                if row == 6:
                    row += 1
                    break
                pos = self.current_board[row, column]
                row += 1
            if player == 0:
                self.current_board[row-2, column] = "O"
            elif player == 1:
                self.current_board[row-2, column] = "X"


def initGridPlayer():
    state = np.zeros((6,7,2))
    return state


def check_winner(state,player,ccount):
    if player == 0:
        if(ccount == 42):
            return 2
        for row in range(6):
            for col in range(7):
                if not np.array_equal(state[row, col],np.array([0,0])):
                    # rows
                    try:
                        if np.array_equal(state[row, col],np.array([0,1])) and np.array_equal(state[row + 1, col],np.array([0,1])) and \
                            np.array_equal(state[row + 2, col],np.array([0,1])) and np.array_equal(state[row + 3, col],np.array([0,1])):
                            print("Column at ", col)
                            return -1
                    except IndexError:
                        next
                    # columns
                    try:
                        if np.array_equal(state[row, col],np.array([0,1])) and np.array_equal(state[row, col + 1],np.array([0,1])) and \
                            np.array_equal(state[row, col + 2],np.array([0,1])) and np.array_equal(state[row, col + 3],np.array([0,1])):
                            print("Row at ", row)
                            return -1
                    except IndexError:
                        next
                    # \ diagonal
                    try:
                        if np.array_equal(state[row, col],np.array([0,1])) and np.array_equal(state[row + 1, col + 1],np.array([0,1])) and \
                            np.array_equal(state[row + 2, col + 2],np.array([0,1])) and np.array_equal(state[row + 3, col + 3],np.array([0,1])):
                            print("Diagonal at ", row+3, ",", col+3)
                            return -1
                    except IndexError:
                        next
                    # / diagonal
                    try:
                        if np.array_equal(state[row, col],np.array([0,1])) and np.array_equal(state[row + 1, col - 1],np.array([0,1])) and \
                            np.array_equal(state[row + 2, col - 2],np.array([0,1])) and np.array_equal(state[row + 3, col - 3],np.array([0,1]))\
                            and (col-3) >= 0:
                            print("Diagonal at ", row+3, ",", col-3)
                            return -1
                    except IndexError:
                        next

        return 0

    if player == 1:
        if(ccount == 42):
            return 2
        for row in range(6):
            for col in range(7):
                if not np.array_equal(state[row, col],np.array([0,0])):
                    # rows
                    try:
                        if np.array_equal(state[row, col],np.array([1,0])) and np.array_equal(state[row + 1, col],np.array([1,0])) and \
                            np.array_equal(state[row + 2, col],np.array([1,0])) and np.array_equal(state[row + 3, col],np.array([1,0])):
                            print("Column at ", col)
                            return 1
                    except IndexError:
                        next
                    # columns
                    try:
                        if np.array_equal(state[row, col],np.array([1,0])) and np.array_equal(state[row, col + 1],np.array([1,0])) and \
                            np.array_equal(state[row, col + 2],np.array([1,0])) and np.array_equal(state[row, col + 3],np.array([1,0])):
                            print("Row at ", row)
                            return 1
                    except IndexError:
                        next
                    # \ diagonal
                    try:
                        if np.array_equal(state[row, col],np.array([1,0])) and np.array_equal(state[row + 1, col + 1],np.array([1,0])) and \
                            np.array_equal(state[row + 2, col + 2],np.array([1,0])) and np.array_equal(state[row + 3, col + 3],np.array([1,0])):
                            print("Diagonal at ", row+4, ",", col+2)
                            return 1
                    except IndexError:
                        next
                    # / diagonal
                    try:
                        if np.array_equal(state[row, col],np.array([1,0])) and np.array_equal(state[row + 1, col - 1],np.array([1,0])) and \
                            np.array_equal(state[row + 2, col - 2],np.array([1,0])) and np.array_equal(state[row + 3, col - 3],np.array([1,0]))\
                            and (col-3) >= 0:
                            print("Diagonal at ", row+3, ",", col-3)
                            return 1
                    except IndexError:
                        next
        return 0


def makeMove(state, column, player):
    if not np.array_equal(state[0, column],np.array([0,0])):
            return "Invalid move"
    else:
        zarr = np.array([0,0])
        row = 0; pos = np.array([0,0])
        while (np.array_equal(pos, zarr)):
            if row == 6:
                row += 1
                break
            pos = state[row, column]
            row += 1
        if player == 0:
            state[row-2, column] = np.array([0,1])
        elif player == 1:
            state[row-2, column] = np.array([1,0])
    return state


def getReward(state,player,ccount):
    w = check_winner(state,player,ccount)
    '''if(w==0):
        rewards=-2
    elif(w==1 or w==-1):
        rewards=1
    elif(w==2):
        rewards=-2
    return rewards'''

    if(w == 1):
        if(player == 1):
            rewards = 1
        else:
            rewards = -2
    elif(w == -1):
        if(player == -1):
            rewards = 1
        else:
            rewards = -2
    elif(w == 2):
        rewards = 0.5
    else:
        rewards = -2
    return rewards



def checkActions(state): # returns all possible moves
        acts = []
        for col in range(7):
            if np.array_equal(state[0, col],np.array([0,0])):
                acts.append(col)
        return acts


def Vpos(state, qval): #Checks if position with max Qval is valid
    while(1):
        action1=(np.argmax(qval))
        if not np.array_equal(state[0,action1],np.array([0,0])):
            qval[0][action1]=-1000
        else:
            break
    return action1   


def train_keras():
    model = Sequential()
    
    model.add(Dense(160, init='lecun_uniform', input_shape=(84,)))
    model.add(Activation('relu'))
    
    model.add(Dense(80, init='lecun_uniform'))
    model.add(Activation('relu'))
    
    model.add(Dense(7, init='lecun_uniform'))
    model.add(Activation('linear')) 
    
    rms = Adam()
    model.compile(loss='mse', optimizer=rms)
    
    
    epochs = 1000
    gamma = 0.9 
    epsilon = 1
    cval = 0
    for i in range(epochs):
        print("Epoch: ",i)
        turn = 0
        state = initGridPlayer()
        status = 1
        
        if(status == 1):
            player = turn%2
            qval = model.predict(state.reshape(1,84), batch_size=1)
            actions = checkActions(state)            
            if (random.random() < epsilon): 
                action = random.choice(actions)
            else: 
                action = (np.argmax(qval))
            new_state = makeMove(state, action, player)
            cval += 1
            reward = getReward(new_state,player,cval)
            newQ = model.predict(new_state.reshape(1,84), batch_size=1)
            maxQ = np.max(newQ)
            y = np.zeros((1,7))
            y[:] = qval[:]
            if reward == -2: #non-terminal state
                update = (reward + (gamma * maxQ))
            else: #terminal state
                update = reward
            y[0][action] = update #target output
            model.fit(state.reshape(1,84), y, batch_size=1)
            state = new_state
            if reward != -2:
                status = 0
        if epsilon > 0.1:
            epsilon -= (1/epochs) # decrease the probability of exploring
        turn+=1
    
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

def test_keras():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")
    print()
    print()
    print()
    state = initGridPlayer()
    cb = board()
    ccount = 0 
    print(cb.current_board)
    player = 0
    while(1):
        if(player == 0):   
            print()          
            print("Player O")
            c=int(input("Enter column "))
            #print(state[0, c])
            while(not(np.array_equal(state[0, c],np.array([0,0])))):
                c=int(input("Invalid! Enter again "))
            cb.drop_piece(c,player)
            #print("state before dropping",state)
            new_state = makeMove(state, c, player)
            ccount+=1
            #print("return state",new_state)
            state = new_state
            #print("state after dropping",state)
            print(cb.current_board)
            cw = check_winner(state,player,ccount)
            if(cw==-1):
                print(" O WINS!")
                print()
                print()
                break
            elif(cw==2):
                print(" It's a DRAW")
                print()
                print()
                break
            player = 1
        elif(player == 1):
            print()
            print("Player X")
            qval = model.predict(state.reshape(1,84), batch_size=1)
            time.sleep(0.5)
            action=Vpos(state,qval)
            new_state = makeMove(state, action, player)
            state = new_state
            cb.drop_piece(action,player)
            ccount+=1
            print(cb.current_board)
            cw = check_winner(state,player,ccount)
            if(cw==1):
                print(" X WINS!")
                print()
                print()
                break
            elif(cw==2):
                print(" It's a DRAW ")
                print()
                print()
                break
            player = 0

def two_player_keras():
    state = initGridPlayer()
    cb = board()
    print(cb.current_board)
    ccount = 0
    while(1):
        player = 0
        print("Player O")
        c=int(input("Enter column "))
        cb.drop_piece(c,player)
        new_state = makeMove(state, c, player)
        state = new_state
        ccount += 1
        print(cb.current_board)
        cw = check_winner(state,player,ccount)
        if(cw==-1):
            print(" O WINS!")
            print()
            break
        elif(cw==2):
            print(" It's a DRAW")
            print()
            break
        print()

        player = 1
        print("Player (X)")
        c=int(input("Enter column "))
        cb.drop_piece(c,player)
        new_state = makeMove(state, c, player)
        state = new_state
        ccount += 1
        print(cb.current_board)
        cw = check_winner(state,player,ccount)
        if(cw==1):
            print(" X WINS!")
            print()
            break
        elif(cw==2):
            print(" It's a DRAW")
            print()
            break
        print()
        
def ai_ai_keras():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    state = initGridPlayer()
    cb = board()
    ccount = 0
    print(cb.current_board)
    status = 1
    player = 0
    while(1):
        if(player == 0):   
            print()
            print("Player O")
            time.sleep(0.5)
            qval = model.predict(state.reshape(1,84), batch_size=1)
            action=Vpos(state,qval)
            new_state = makeMove(state, action, player)
            state = new_state
            ccount += 1
            cb.drop_piece(action,player)
            print(cb.current_board)
            cw = check_winner(state,player,ccount)
            if(cw==-1):
                print(" O WINS!")
                print()
                print()
                break
            elif(cw==2):
                print(" It's a DRAW")
                print()
                print()
                break
            player = 1

        elif(player == 1):
            print()
            print("Player X")
            time.sleep(0.5)
            qval = model.predict(state.reshape(1,84), batch_size=1)
            action=Vpos(state,qval)
            new_state = makeMove(state, action, player)
            state = new_state
            cb.drop_piece(action,player)
            ccount += 1
            print(cb.current_board)
            cw = check_winner(state,player,ccount)
            if(cw==1):
                print(" X WINS!")
                print()
                print()
                break
            elif(cw==2):
                print(" It's a DRAW")
                print()
                print()
                break
            player = 0

'''
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

#If you want to read
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = keras.models.model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")
'''