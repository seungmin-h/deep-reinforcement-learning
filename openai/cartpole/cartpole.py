#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:03:00 2020

@author: seungmin
"""
import numpy as np
import gym
import matplotlib.pyplot as plt

from matplotlib import animation


def display_frames_as_gif(frames):
    
    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0),
               dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames),
                                   interval=50)

    anim.save('movie_cartpole.mp4')
    
    
frames=[]
env = gym.make('CartPole-v0')
observation = env.reset()

for step in range(200):
    frames.append(env.render(mode='rgb_array'))
    action = np.random.choice(2) 
    observation, reward, done, info = env.step(action)
    print(observation, reward, done, info)
    
    
print("Saving all frames to .mp4 format...")
display_frames_as_gif(frames)
    
env.close()
