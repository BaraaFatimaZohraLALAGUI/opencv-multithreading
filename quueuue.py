import multiprocessing as mp
from typing import Any

class RetrievalQueues:
    def __init__(self, maxsize: int = 0, ctx: Any = None) -> None:
        if ctx is None:
            ctx = mp.get_context()
        self.maxsize = maxsize
        self.queue = ctx.Queue(maxsize)  
        self.list = mp.Manager().list ()

    def __getitem__(self, key):
            # print('HERE -----------', len(self.list))
            return self.list[key]
        # else:
        #     raise IndexError("Index out of range")

    def get(self, block=True, timeout=None):
        item = self.queue.get(block, timeout)
        del self.list[0]
        return item

    def put(self, item):
        if self.queue.qsize() < self.maxsize:
            self.queue.put(item)
            self.list.append(item) 


    def qsize(self):
        return self.queue.qsize() 
