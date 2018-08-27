from functions import *

# class to store data from video
class VehicleDetectVideo():
    def __init__(self,n):
        self.n = n
        self.prev_rects = [] 
        
    def addPreds(self, rects):
        self.prev_rects.append(rects)
        if len(self.prev_rects) > self.n:
            self.prev_rects = self.prev_rects[len(self.prev_rects)-self.n:] # throw out oldest rectangle set(s)
            

