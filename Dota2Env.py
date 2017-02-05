import sys
from threading import Thread
from time import sleep
import numpy as np
from dota2comm import Dota2Comm
import queue

ActionMap = {
    "laning": np.asarray([1, 0, 0]),
    "attack": np.asarray([0, 1, 0]),
    "retreat": np.asarray([0, 0, 1]),
}

#Helper function
def ParseLine(ll):
    ll = ll.split(" ")
    return np.asarray([float(s) for s in ll[1:24]]),float(ll[24]),ActionMap[ll[25]]

class Dota2Env():

    def __init__(self, name):
        self.name = name
        self.dota = Dota2Comm(name)
        self.StateQueue = queue.Queue()
        self.OrderQueue = queue.Queue()

        self.StartThreads()

    def WaitDota2Msg(self):
        while self.running:
            msg = self.dota.receiveMessage()
            if msg is not None:
                self.StateQueue.put(msg)

    def SendDota2Msg(self):
        while self.running:
            order = self.OrderQueue.get()
            msg = str(np.argmax(order[0]))
            self.dota.sendMessage(msg)

    def StartThreads(self):
        self.running = True
        self.threadrecv = Thread(target=self.WaitDota2Msg)
        self.threadrecv.start()
        self.threadsend = Thread(target=self.SendDota2Msg)
        self.threadsend.start()

    def GiveOrder(self,order):
        self.OrderQueue.put(order)
    
    def GetStateRewardAction(self):
        origin_str = self.StateQueue.get()
        return ParseLine(origin_str)
    
    def Step(self, order):
        self.GiveOrder(order)
        return self.GetStateRewardAction()

    def Stop(self):
        self.running = False
        self.threadrecv.join()
        self.threadsend.join()
