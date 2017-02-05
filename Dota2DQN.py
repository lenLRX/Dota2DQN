import numpy as np
import AC_Network

import tensorflow as tf

import sys
from threading import Thread
from time import sleep

from Dota2Env import Dota2Env
import queue

import ReplayBuffer

class TrainingTask():
    def __init__(self, name, net,
     buffer, env, session, BATCH_SIZE, GAMMA):
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.name = name
        self.net = net
        self.buffer = buffer
        self.env = env
        self.session = session
        self.s_t, _, _ = self.env.GetStateRewardAction()
    
    def step(self, i):
        self.a_t, self.v_t = self.session.run(
            [self.net.policy, self.net.value],
            feed_dict={
                self.net.inputs: [self.s_t]
            }
        )

        

        self.s_t1, self.r_t, self.action_hot = self.env.Step(self.a_t)
        self.buffer.put([self.s_t, self.action_hot, 
            self.r_t, self.s_t1, self.v_t[0][0]])
        
        print([self.s_t, self.action_hot, 
            self.r_t, self.s_t1, self.v_t],self.a_t)

        if i < 10:
            return
        
        minibatch = self.buffer.get(self.BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        value_batch = np.asarray([data[4] for data in minibatch] + [self.v_t[0][0]])
        advantages = reward_batch + self.GAMMA * value_batch[1:] - value_batch[:-1]

        v_l, p_l, e_l, g_n, v_n, _, responsible_out = self.session.run(
            [
                self.net.value_loss,
                self.net.policy_loss,
                self.net.entropy,
                self.net.grad_norms,
                self.net.var_norms,
                self.net.apply_grads,
                self.net.responsible_outputs
            ],
            feed_dict={
                self.net.inputs: state_batch,
                self.net.actions: action_batch,
                self.net.advantages: advantages.reshape((len(minibatch),1)),
                self.net.target_v: reward_batch.reshape((len(minibatch),1))
            }
        )
        print(self.name,"i:", i, v_l / len(minibatch),
              p_l / len(minibatch), e_l / len(minibatch), g_n, v_n)
        
        self.s_t = self.s_t1

def run():
    GAMMA = 0.9
    BATCH_SIZE = 1000
    BUFFER_SIZE = 10000
    action_dim = 3
    state_dim = 23
    hidden_unit = 100
    Env_radiant = Dota2Env("SFradiant")
    Env_dire = Dota2Env("SFdire")
    AC_net = AC_Network.AC_Network(state_dim, action_dim, hidden_unit)
    
    buffer = ReplayBuffer.ReplayBuffer(BUFFER_SIZE)

    session = tf.InteractiveSession()
    session.run(tf.initialize_all_variables())

    radiant_task = TrainingTask("SFradiant", AC_net, buffer,
        Env_radiant, session, BATCH_SIZE, GAMMA)
    
    dire_task = TrainingTask("SFdire", AC_net, buffer,
        Env_dire, session, BATCH_SIZE, GAMMA)
    
    for i in range(100000000):
        radiant_task.step(i)
        dire_task.step(i)

'''
    s_t, _, _ = Env.GetStateRewardAction()

    for i in range(100000000):
        a_t, v_t = session.run(
            [AC_net.policy, AC_net.value],
            feed_dict={
                AC_net.inputs: [s_t]
            }
        )

        print("state", s_t)

        print("action", a_t, "value", v_t)

        s_t1, r_t, action_hot = Env.Step(a_t)
        buffer.put([s_t, action_hot, r_t, s_t1, v_t[0][0]])

        if i < 10:
            continue
        minibatch = buffer.get(BATCH_SIZE)

        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        value_batch = np.asarray([data[4] for data in minibatch] + v_t)

        advantages = reward_batch + GAMMA * value_batch[1:] - value_batch[:-1]
        #print("value_batch", value_batch)
        #print("reward_batch", reward_batch)
        #print("advantages", advantages)

        v_l, p_l, e_l, g_n, v_n, _, responsible_out = session.run(
            [
                AC_net.value_loss,
                AC_net.policy_loss,
                AC_net.entropy,
                AC_net.grad_norms,
                AC_net.var_norms,
                AC_net.apply_grads,
                AC_net.responsible_outputs
            ],
            feed_dict={
                AC_net.inputs: state_batch,
                AC_net.actions: action_batch,
                AC_net.advantages: advantages.reshape((len(minibatch),1)),
                AC_net.target_v: reward_batch.reshape((len(minibatch),1))
            }
        )

        #print("responsible_outputs", responsible_out)

        print("i:", i, v_l / len(minibatch),
              p_l / len(minibatch), e_l / len(minibatch), g_n, v_n)
        
        s_t = s_t1

'''

if __name__ == "__main__":
    run()
