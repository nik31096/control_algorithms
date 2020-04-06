#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import numpy


# In[2]:


env = gym.make('MountainCar-v0')


# In[ ]:


class Node:
    def __init__(self, parent):
        self.parent = parent
        
    def

