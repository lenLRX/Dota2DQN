import numpy as np
import AC_Network

import tensorflow as tf

import sys
from threading import Thread
from time import sleep

from Dota2Env import Dota2Env
import queue

import ReplayBuffer


def run():
    GAMMA = 0.9
    BATCH_SIZE = 1000
    BUFFER_SIZE = 10000
    action_dim = 3
    state_dim = 22
    Env = Dota2Env()
    AC_net = AC_Network.AC_Network(state_dim, action_dim, 100)
    
    buffer = ReplayBuffer.ReplayBuffer(BUFFER_SIZE)

    session = tf.InteractiveSession()
    session.run(tf.initialize_all_variables())

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
        value_batch = np.asarray([data[4] for data in minibatch])

        advantages = reward_batch - value_batch
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

if __name__ == "__main__":
    run()
