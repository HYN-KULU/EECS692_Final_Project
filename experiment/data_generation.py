from mypolicy import MyPolicy_CL
from metaworld_exp.utils import get_seg, get_cmat, collect_video, sample_n_frames
import sys
sys.path.append('core')
import imageio.v2 as imageio
import numpy as np
from myutils import get_flow_model, pred_flow_frame, get_transforms, get_transformation_matrix
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as env_dict
from metaworld import policies
from tqdm import tqdm
import cv2
import imageio
import json
import os
from flowdiffusion.inference_utils import get_video_model, pred_video
import random
import torch
from argparse import ArgumentParser

def get_task_text(env_name):
    name = " ".join(env_name.split('-')[:-3])
    return name

def get_policy(env_name):
    name = "".join(" ".join(get_task_text(env_name)).title().split(" "))
    name=name[0].upper()+name[1:].lower()
    policy_name = "Sawyer" + name + "V2Policy"    
    try:
        policy = getattr(policies, policy_name)()
    except:
        policy = None
    return policy


if __name__=="__main__":
    # print(env_dict.keys())
    env_name="assembly-v2-goal-observable"
    benchmark_env = env_dict[env_name]
    task_name=get_task_text(env_name)
    env = benchmark_env(seed=10)
    obs = env.reset()
    policy=get_policy(env_name)
    env_list=[]
    images=np.zeros([8,3,240,320])
    for i in range(100):
        action=policy.get_action(obs)
        # action=np.array([-1,1,1,1])
        obs, reward, done, info = env.step(action)
        done = info['success']
        image, depth = env.render(depth=True, offscreen=True, camera_name='corner', resolution=(320, 240)) # image shape: 240,320,3
        image=image.transpose(2,0,1)
        # if(i%5==0):
        #     images[i//5]=image
        # if done:
            # break
    # print(info['success'])
    print(image.shape)
    print(len(images))
    torch.save(images,"./images.pth")
    print(obs)
