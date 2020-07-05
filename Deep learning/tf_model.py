# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 18:42:08 2020

@author: Sanjana moudgalya
"""

import random
import numpy as np
import os
import math
import tensorflow as tf

################################################ 

RED_PLAYER = 'R'
YELLOW_PLAYER = 'Y'
RED_PLAYER_VAL = -1
YELLOW_PLAYER_VAL = 1
EMPTY = ' '
EMPTY_VAL = 0
HORIZONTAL_SEPARATOR = ' | '
GAME_STATE_X = -1
GAME_STATE_O = 1
GAME_STATE_DRAW = 0
GAME_STATE_NOT_ENDED = 2
VERTICAL_SEPARATOR = '__'
NR = 6
NC = 7
SEQ = 4

size_input = 42 
size_hidden1 = 84 
size_hidden2 = 168
size_hidden3 = 336
size_hidden4 = 504
size_hidden5 = 672
size_output = 7

X = tf.placeholder(dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32)

################################################ 

def createBoard():
    board = [
            [EMPTY_VAL, EMPTY_VAL, EMPTY_VAL, EMPTY_VAL, EMPTY_VAL, EMPTY_VAL, EMPTY_VAL],
            [EMPTY_VAL, EMPTY_VAL, EMPTY_VAL, EMPTY_VAL, EMPTY_VAL, EMPTY_VAL, EMPTY_VAL],
            [EMPTY_VAL, EMPTY_VAL, EMPTY_VAL, EMPTY_VAL, EMPTY_VAL, EMPTY_VAL, EMPTY_VAL],
            [EMPTY_VAL, EMPTY_VAL, EMPTY_VAL, EMPTY_VAL, EMPTY_VAL, EMPTY_VAL, EMPTY_VAL],
            [EMPTY_VAL, EMPTY_VAL, EMPTY_VAL, EMPTY_VAL, EMPTY_VAL, EMPTY_VAL, EMPTY_VAL],
            [EMPTY_VAL, EMPTY_VAL, EMPTY_VAL, EMPTY_VAL, EMPTY_VAL, EMPTY_VAL, EMPTY_VAL]
        ]
    return board

################################################ 

def printBoard(board):
    for i in range(len(board)):
        for j in range(len(board[i])):
            print (VERTICAL_SEPARATOR, end='')
        print (os.linesep)
        for j in range(len(board[i])):
            if RED_PLAYER_VAL == board[i][j]:
                print(RED_PLAYER, end='')
            elif YELLOW_PLAYER_VAL == board[i][j]:
                print(YELLOW_PLAYER, end='')
            elif EMPTY_VAL == board[i][j]:
                print(EMPTY, end='')
            print(HORIZONTAL_SEPARATOR, end='')
        print (os.linesep)
        for j in range(len(board[i])):
            print(VERTICAL_SEPARATOR, end='')
    print (os.linesep)
 
################################################ 

def printWinner(win):
    if(win==1):
        print("Yellow wins the game!!")
    elif(win==-1):
        print("Red wins the game!!")
    else:
        print("This game is a draw.")
    
################################################
    
def getAvailableMoves(board):
    availableMoves = []
    for j in range(NC):
        if board[NR - 1][j] == EMPTY_VAL:
            availableMoves.append([NR - 1, j])
        else:
            for i in range(NR - 1):
                if board[i][j] == EMPTY_VAL and board[i + 1][j] != EMPTY_VAL:
                    availableMoves.append([i, j])
    return availableMoves
       
##################################################

def getGameResult(board):
    winnerFound = False
    currentWinner = None
    # Find winner on horizontal
    for i in range(NR):
        for j in range(NC - SEQ + 1):
            if not winnerFound:
                if board[i][j] != 0 and board[i][j] == board[i][j+1] and board[i][j] == board[i][j + 2] and \
                        board[i][j] == board[i][j + 3]:
                    currentWinner = board[i][j]
                    #print("a %s",currentWinner)
                    winnerFound = True

    # Find winner on vertical
    if not winnerFound:
        for j in range(NC):       
            for i in range(NR - SEQ + 1):
                if not winnerFound:
                    if board[i][j] != 0 and board[i][j] == board[i+1][j] and board[i][j] == board[i+2][j] and \
                            board[i][j] == board[i+3][j]:
                        currentWinner = board[i][j]
                        #print("b %s",currentWinner)
                        winnerFound = True

    # Check lower left diagonals
    if not winnerFound:
        for i in range(NR - SEQ + 1): #1
           for j in range(NC - SEQ + 1):
               if board[i][j] != 0 and board[i][j] == board[i + 1][j + 1] and board[i][j] == board[i + 2][j + 2] and \
                       board[i][j] == board[i + 3][j + 3]:
                   currentWinner = board[i][j]
                   #print("c %s",currentWinner)
                   winnerFound = True

    # Check upper right diagonals
    if not winnerFound:
        for i in range(NR - SEQ + 1):
            for j in range(3,NC):
                if board[i][j] != 0 and board[i][j] == board[i + 1][j - 1] and board[i][j] == board[i + 2][j - 2] and \
                        board[i][j] == board[i + 3][j - 3]:
                    currentWinner = board[i][j]
                    #print("d %s",currentWinner)
                    winnerFound = True

    if winnerFound: 
        #print("e %s",currentWinner)
        return currentWinner
    else:
        drawFound = True
        # Check for draw
        for i in range(NR):
            for j in range(NC):
                if board[i][j] == EMPTY_VAL:
                    drawFound = False
        if drawFound:
            return GAME_STATE_DRAW
        else:
            return GAME_STATE_NOT_ENDED
        
######################################################

def makeMove(board, action,player):
    j=action
    for k in range(NR):
        if not board[k][j]:
            i=k
            break
    board[i][j]=player
    return board

################################################ 

def getReward(board,player):
    win=getGameResult(board)
    if(win==0):
        reward=0.5
    elif(win==1 or win==-1):
        #reward=1
        if(win==player):
            reward=1
        else:
            reward=-10
    elif(win==2):
        reward=-2
    return(reward,win)
 
################################################

def checkAction(board,qval):
    action1=(np.argmax(qval))
    while(board[5][action1]!=0):
        qval[0][action1]=np.min(qval)-1
        action1=(np.argmax(qval))
    return action1

################################################            

limit1 = math.sqrt(3/size_input)
seed1 = np.random.uniform(-limit1,limit1)
wh1 = tf.Variable(tf.random_normal([size_input,size_hidden1],seed=seed1))
bh1 = tf.Variable(tf.random_normal([1,size_hidden1],seed=seed1))

limit2 = math.sqrt(3/size_hidden1)
seed2 = np.random.uniform(-limit2,limit2)
wh2 = tf.Variable(tf.random_normal([size_hidden1,size_hidden2],seed=seed2))
bh2 = tf.Variable(tf.random_normal([1,size_hidden2],seed=seed2))

limit3 = math.sqrt(3/size_hidden2)
seed3 = np.random.uniform(-limit3,limit3)
wh3 = tf.Variable(tf.random_normal([size_hidden2,size_hidden3],seed=seed3))
bh3 = tf.Variable(tf.random_normal([1,size_hidden3],seed=seed3))

limit4 = math.sqrt(3/size_hidden3)
seed4 = np.random.uniform(-limit4,limit4)
wh4 = tf.Variable(tf.random_normal([size_hidden3,size_hidden4],seed=seed4))
bh4 = tf.Variable(tf.random_normal([1,size_hidden4],seed=seed4))

limit5 = math.sqrt(3/size_hidden4)
seed5 = np.random.uniform(-limit5,limit5)
wh5 = tf.Variable(tf.random_normal([size_hidden4,size_hidden5],seed=seed5))
bh5 = tf.Variable(tf.random_normal([1,size_hidden5],seed=seed5))

limit6 = math.sqrt(3/size_hidden5)
seed6 = np.random.uniform(-limit6,limit6)
wo = tf.Variable(tf.random_normal([size_hidden5, size_output],seed=seed6))
bo = tf.Variable(tf.random_normal([1,size_output],seed=seed6))

net_h1 = tf.matmul(X, wh1) + bh1
act_h1 = tf.nn.relu(net_h1)

net_h2 = tf.matmul(act_h1, wh2) + bh2
act_h2 = tf.nn.relu(net_h2)

net_h3 = tf.matmul(act_h2, wh3) + bh3
act_h3 = tf.nn.relu(net_h3)

net_h4 = tf.matmul(act_h3, wh4) + bh4
act_h4 = tf.nn.relu(net_h4)

net_h5 = tf.matmul(act_h4, wh5) + bh5
act_h5 = tf.nn.relu(net_h5)

net_o = tf.matmul(act_h5, wo) + bo
pred = tf.nn.sigmoid(net_o)

loss = tf.reduce_mean(tf.squared_difference(pred, Y))
opt = tf.train.AdamOptimizer(learning_rate=0.0000000000000001).minimize(loss)

################################################ 

def train_tf():
    #define parameters to be learnt 
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    epochs = 1000
    gamma = 0.9 
    epsilon = 1
    with tf.Session() as sess:
        sess.run(init)    
        # train the model, that is find x to minimize loss
        for i in range(epochs):
            board = createBoard()
            #print(board)
            status = 1
            #while game still in progress
            turn=1
            while(status == 1):
                player=math.pow(-1,turn)
                qval,allq= sess.run([pred,net_o],feed_dict={X:np.reshape(board,(1,42))}) #for nomal gradient descent
                # loop for stocastic gradient descent
                actions = getAvailableMoves(board)            
                if (random.random() < epsilon): #choose random action (exploring)
                    action = random.choice(actions)
                    action = action[1]
                else: #choose best action from Q(s,a) values (exploiting)
                    action = checkAction(board,qval)
                board = makeMove(board, action, player) #make a new move
                reward,win = getReward(board,player) #get the reward
                newQ,abc = sess.run([pred,net_o],feed_dict={X:np.reshape(board,(1,42))})
                maxQ = np.max(newQ)
                y = np.zeros((1,7))
                y[:] = qval[:]
                if reward == -2: #non-terminal state
                    update = (reward + (gamma * maxQ))
                else: #terminal state
                    update = reward
                y[0][action] = update #target output
                sess.run([opt,wh1],feed_dict={X:np.reshape(board,(1,42)) , Y:y})
                if reward != -2:
                    status = 0
                    #print(win)
                turn=turn+1
            if epsilon > 0.1:
                epsilon -= (1/epochs)
            print("epoch:",i ,"Loss:", loss.eval(feed_dict={X:np.reshape(board,(1,42)) , Y:y}))
        saver.save(sess, "model.ckpt")
        
################################################ 

def test_tf(saver):  
    with tf.Session() as sess:
        saver.restore(sess, "model.ckpt")
        i=0
        board = createBoard()
        print("Initial State:")
        printBoard(board)
        status = 1
        while(status == 1):
            player=math.pow(-1,i)
            if(player==1):             
                print("Enter the position:")
                pos= int(input())
                board = makeMove(board, pos,player)
            else:
                qval,allq= sess.run([pred,net_o],feed_dict={X:np.reshape(board,(1,42))})
                action1=checkAction(board,qval)
                board = makeMove(board, action1,player)
                printBoard(board)
                reward,win = getReward(board,player)
                if reward != -2:
                    status = 0
                    printWinner(win)
            i+=1
        
################################################ 

def ai_ai_tf(saver):
     with tf.Session() as sess:
        saver.restore(sess, "model.ckpt")
        i=0
        board = createBoard()
        print("Initial State:")
        printBoard(board)
        status = 1
        while(status == 1):
            player=math.pow(-1,i)
            qval,allq= sess.run([pred,net_o],feed_dict={X:np.reshape(board,(1,42))})
            action1=checkAction(board,qval)
            board = makeMove(board, action1,player)
            printBoard(board)
            reward,win = getReward(board,player)
            if reward != -2:
                status = 0
                printWinner(win)
            i += 1

################################################

def two_player_tf():
    i=0
    board = createBoard()
    print("Initial State:")
    printBoard(board)
    status = 1
    while(status == 1):
        player=math.pow(-1,i) 
        print("Enter the position :")
        pos= int(input())
        board = makeMove(board,pos,player)
        printBoard(board)
        reward,win = getReward(board,player)
        if reward != -2:
            status = 0
            printWinner(win)
        i += 1
        
################################################        