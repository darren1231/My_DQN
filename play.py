# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 12:50:51 2016

@author: darren
"""

#!/usr/bin/env python

import tensorflow as tf
import cv2
import sys
import datetime
sys.path.append("Wrapped Game Code/")
import pong_fun as game# whichever is imported "as game" will be used
import tetris_fun
import random
import numpy as np
import os
from collections import deque

import brain as net

GAME = 'pong' # the name of the game being played for log files
ACTIONS = 3 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 500. # timesteps to observe before training
EXPLORE = 500. # frames over which to anneal epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 5000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
K = 1 # only select an action every Kth frame, repeat prev for others

######################################################################
start = datetime.datetime.now()
store_network_path="temp/my_networks/"
tensorboard_path = "temp/logs/"
out_put_path = "temp/logs_" + GAME

if os.path.exists('temp'):
    pass
else:
    os.makedirs(store_network_path)
    os.makedirs(out_put_path)

pretrain_number=0


######################################################################

def sencond2time(senconds):

	if type(senconds)==type(1):
		h=senconds/3600
		sUp_h=senconds-3600*h
		m=sUp_h/60
		sUp_m=sUp_h-60*m
		s=sUp_m
		return ",".join(map(str,(h,m,s)))
	else:
		return "[InModuleError]:sencond2time(senconds) invalid argument type"

    
    

def trainNetwork(s, readout,sess):

    
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.mul(readout, a), reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # printing
    a_file = open(out_put_path  + "/readout.txt", 'w')
    h_file = open(out_put_path  + "/hidden.txt", 'w')


    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state(store_network_path)
    
    #saver.restore(sess, "new_networks/pong-dqn-"+str(pretrain_number))    
    
#    if checkpoint and checkpoint.model_checkpoint_path:
#        saver.restore(sess, checkpoint.model_checkpoint_path)
#	#saver.restore(sess, "my_networks/pong-dqn-26000")
#        print "Successfully loaded:", checkpoint.model_checkpoint_path
#    else:
#        print "Could not find old network weights"
    
    print "Press any key and Enter to continue:"
    raw_input()

    epsilon = INITIAL_EPSILON
    t = 0
    total_score=0
    positive_score=0
    while True:
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict = {s : [s_t]})[0]
        
        

        a_t = np.zeros([ACTIONS])
        action_index = 0
        if random.random() <= epsilon or t <= OBSERVE:
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        for i in range(0, K):
            # run the selected action and observe next state and reward
            x_t1_col, r_t, terminal = game_state.frame_step(a_t)
            x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1,1,255,cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            s_t1 = np.append(x_t1, s_t[:,:,0:3], axis = 2)

            # store the transition in D
            D.append((s_t, a_t, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()
                
        total_score=total_score+r_t;
        if r_t==1:
            positive_score=positive_score+r_t

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                # if terminal only equals reward
                if minibatch[i][4]:
                    y_batch.append(r_batch[i])
                else:
                    #y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))                   
                    

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch})

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 1000 == 0:
            saver.save(sess, store_network_path + GAME + '-dqn', global_step = t+pretrain_number)
            
            #saver.save(sess, 'new_networks/' + GAME + '-dqn', global_step = t)

        if t % 500 == 0:  
            now=datetime.datetime.now()
            diff_seconds=(now-start).seconds
            time_text=sencond2time(diff_seconds)
            
#            result = sess.run(merged,feed_dict = {s : [s_t]})
#            writer.add_summary(result, t+pretrain_number)
            a_file.write(str(t+pretrain_number)+','+",".join([str(x) for x in readout_t]) + \
            ','+str(total_score)+ ','+str(positive_score) \
            +','+time_text+'\n')

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        print "TIMESTEP:", t+pretrain_number, "/ ACTION:", action_index, "/ REWARD:", r_t, "/ Q_MAX: %e" % np.max(readout_t),'  time:(H,M,S):' \
        + sencond2time((datetime.datetime.now()-start).seconds)
        print 'Total score:',total_score,' Positive_score:',positive_score,'   up:',readout_t[0],'    down:',readout_t[1],'  no:',readout_t[2]
       
        # write info to files
        
        #if t % 10000 <= 100:
            #a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            #h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            #cv2.imwrite("logs_pong/frame" + str(t) + ".png", x_t1)
        

def playGame():
    sess = tf.InteractiveSession()
    brain_net = net.Brain(3)
    Target_net_brain=net.Brain(3)
    s, readout,variable = brain_net.createNetwork()
    s_new, readout_new,variable_new = brain_net.create_new_Network()
    
    for i in range(len(variable)):
        print variable[i].get_shape()
    #s_T, readout_T,W_conv1_T,b_conv1_T,W_conv2_T,b_conv2_T,W_conv3_T,b_conv3_T,W_fc1_T,b_fc1_T,W_fc2_T,b_fc2_T = Target_net_brain.createNetwork()

#    merged = tf.merge_all_summaries()
#    writer = tf.train.SummaryWriter(tensorboard_path, sess.graph)
    
    trainNetwork(s, readout,sess)
    
def loadnet():
    sess = tf.InteractiveSession()
    brain_net = net.Brain(3)
    s, readout,variable = brain_net.createNetwork()
    
    init = tf.initialize_all_variables()
    sess.run(init)
    
    new_variable_list=sess.run(variable)
    
    print new_variable_list[1]
    
    saver = tf.train.Saver()
    saver.restore(sess, "Old_net/pong-dqn-1320000_train_hit")
        
    old_variable = tf.all_variables()
    old_variable_list=sess.run(old_variable)
    
    
    print old_variable_list[1]
    
    
#    for i in range(0,len(old_variable_list),2):
#        print "w",old_variable_list[i].shape,    "\n"
#        print "b",old_variable_list[i+1].shape,"\n"
    
    for i in range(len(old_variable)):
        assing_op=tf.assign(variable[i],old_variable[i])
        sess.run(assing_op)
    
    new_variable_list=sess.run(variable)
    print new_variable_list[1]
    
    "Implement..........................."   
    game_state = game.GameState()
    
    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)
    
    a_file = open(out_put_path  + "/readout.txt", 'w')
    t=0
    total_score=0
    # get the first state by doing nothing and preprocess the image to 80x80x4
    while True:
        
        readout_t = readout.eval(feed_dict = {s : [s_t]})[0]            
    
        a_t = np.zeros([ACTIONS])
        action_index = 0
        
        action_index = np.argmax(readout_t)
        a_t[action_index] = 1
        
        x_t1_col, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1,1,255,cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:,:,0:3], axis = 2)
    
        s_t = s_t1
        t=t+1
        
        total_score=total_score+r_t;
        
        a_file.write(str(t)+','+",".join([str(x) for x in readout_t]) +','+str(total_score)+'\n')
        
        print "TIMESTEP:", t, "/ ACTION:", action_index, "/ REWARD:", r_t, "/ Q_MAX: %e" % np.max(readout_t)
        print 'Total score:',total_score,'   up:',readout_t[0],'    down:',readout_t[1],'  no:',readout_t[2]
    
    #trainNetwork(s, readout,sess)
#    variable = tf.all_variables()
#    variable_list=sess.run(variable)
def main():
    loadnet()

if __name__ == "__main__":
    main()