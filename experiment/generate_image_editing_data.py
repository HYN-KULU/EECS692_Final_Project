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
from typing import List
# "door-open-v2-goal-observable" "door-close-v2-goal-observable" "basketball-v2-goal-observable" "shelf-place-v2-goal-observable" "button-press-v2-goal-observable" "button-press-topdown-v2-goal-observable" "faucet-close-v2-goal-observable" "faucet-open-v2-goal-observable" "handle-press-v2-goal-observable" "hammer-v2-goal-observable" "assembly-v2-goal-observable"

def get_task_text(env_name):
    name = " ".join(env_name.split('-')[:-3])
    return name

def transform(strings:List[str])->List[str]:
    for i in range(len(strings)):
        strings[i]=strings[i][0].upper()+strings[i][1:]
    return strings        

def get_policy(env_name):
    name = "".join(transform(get_task_text(env_name).split(" ")))
    policy_name = "Sawyer" + name + "V2Policy"   
    try:
        policy = getattr(policies, policy_name)()
    except:
        policy = None
    return policy

def images_to_video(image_path_list, output_path, fps=2):
    with imageio.get_writer(output_path, fps=fps) as writer:
        for image_path in image_path_list:
            if isinstance(image_path, str):  # Check if the path is a string path
                image = imageio.imread(image_path)
            elif isinstance(image_path, np.ndarray):  # Check if it's already an image array
                image = image_path
            else:
                continue  # Skip unknown types
            writer.append_data(image)

def get_images_data(env_name,AVDC_step,seed):
    actions=data['actions']
    images,gap_images=[],[]
    benchmark_env = env_dict[env_name]
    task_name=get_task_text(env_name)
    env = benchmark_env(seed)
    obs = env.reset()
    branch_cache,branch_id,gap,dd=0,0,0,0
    for i in range(500):
        if (i<=AVDC_step):
            action=actions[i]
        else: 
            action,branch_id=policy.get_action(obs)
        obs, reward, done, info = env.step(action)
        done = info['success']
        image, depth = env.render(depth=True, offscreen=True, camera_name='corner', resolution=(320, 240)) # image shape: 240,320,3
        ## This is for illustration. The AVDC model runs trajectory until AVDC_step. Then I use expert policy to run. Every time the model changes a branch, I set it as a subgoal.
        if(i==AVDC_step or (i>AVDC_step and branch_id !=branch_cache)):
            # gap images definition can be seen in the next branch. I only want to add gap images when the model can indeed gets to the next subgoal, otherwise I don't think those data useful.
            images+=gap_images
            gap_images=[]
            images.append(image)
            gap=0
        # If a subgoal takes to long to finish, I save it every 15 steps.
        elif (i>AVDC_step and gap>15):
            gap_images.append(image)
            gap=0
        # branch_cache is used to check whether the branch_id changes from steps to steps.
        branch_cache=branch_id
        # Increment the gap steps
        if(i>AVDC_step):
            gap+=1
        # If done, don't immediately stop, just to see some videos after the goal is finished, it will also be helpful if you see the assembly env.
        if (done):
            if env_name == "button-press-topdown-v2-goal-observable":
                break
            dd+=1
            images+=gap_images
            gap_images=[]
            if (dd>25):
                images+=gap_images
                gap_images=[]
                break
    return images
    

if __name__=="__main__":
    env_name="hammer-v2-goal-observable"
    seed=0
    for seed in range(20,30):
        policy=get_policy(env_name)
        data_trajectory_dir="/nfs/turbo/coe-chaijy/heyinong/image_editing_data/"
        if not os.path.exists(data_trajectory_dir):
            os.makedirs(data_trajectory_dir)
        base_path = f"/nfs/turbo/coe-chaijy/heyinong/results/results_AVDC_mw/videos/{env_name}"
        file_path = os.path.join(base_path, f"corner_{seed}.pth")
        if not os.path.exists(file_path):
            print("here")
            continue
        data=torch.load(file_path)
        image_editing_dir=f"/nfs/turbo/coe-chaijy/heyinong/image_editing_data/{env_name}"
        if not os.path.exists(image_editing_dir):
            os.makedirs(image_editing_dir)
        if os.path.exists(f'{image_editing_dir}/data_{seed}.pth'):
            print("there")
            continue
        image_editing_data_pairs=[]
        for i in range(len(data['actions'])//20):
            print(f"Collecting data for task {env_name} from seed {seed} step {20*i}")
            images=get_images_data(env_name,5*i,seed)
            image_editing_data_pairs+=[(images[j],images[j+1]) for j in range(len(images)-1)]
        for j in range(len(data['plan_steps'])):
            print(f"Collecting data for task {env_name} from seed {seed} step {data['plan_steps'][j]}")
            images=get_images_data(env_name,data['plan_steps'][j],seed)
            image_editing_data_pairs+=[(images[k],images[k+1]) for k in range(len(images)-1)]
        torch.save(image_editing_data_pairs,f'{image_editing_dir}/data_{seed}.pth')
