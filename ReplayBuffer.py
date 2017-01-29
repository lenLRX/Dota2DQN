from collections import deque
import random


class ReplayBuffer:

    def __init__(self, buffersize):
        self.buffer = deque()
        self.buffersize = buffersize

    def put(self, s):
        self.buffer.append(s)
        if(len(self.buffer) > self.buffersize):
            self.buffer.popleft()

    def get(self, batchsize):
        if(len(self.buffer) < batchsize):
            #return random.sample(self.buffer, len(self.buffer))
            return list(self.buffer)
        else:
            #return random.sample(self.buffer, batchsize)
            return list(self.buffer)[len(self.buffer) - batchsize + 1:]
