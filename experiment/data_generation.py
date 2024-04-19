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

if __name__=="__main__":
    env_name="hammer-v2-goal-observable"
    benchmark_env = env_dict[env_name]
    task_name=get_task_text(env_name)
    seed=15
    env = benchmark_env(seed)
    obs = env.reset()
    policy=get_policy(env_name)
    env_list=[]
    images=[]
    fails=[]
    branch_cache=0
    branch_id=0
    gap=0
    base_path = f"/nfs/turbo/coe-chaijy/heyinong/results/results_AVDC_mw/videos/{env_name}"
    file_path = os.path.join(base_path, f"corner_{seed}.pth")
    data=torch.load(file_path)
    actions=data['actions']
    print(data['plan_steps'])
    print(len(actions))
    pred_videos=data['pred_videos'][0]
    pred_images=[]
    for i in range(8):
        print(pred_videos[i].shape)
        pred_images.append(pred_videos[i].transpose(1,2,0))
    images_to_video(pred_images,'./fail_plan.mp4',fps=2)
    # exit()
    failure_step=65
    gap_images=[]
    dd=0
    for i in range(500):
        print(i)
        if (i<=failure_step):
            action=actions[i]
        else: 
            action,branch_id=policy.get_action(obs)
        obs, reward, done, info = env.step(action)
        done = info['success']
        image, depth = env.render(depth=True, offscreen=True, camera_name='corner', resolution=(320, 240)) # image shape: 240,320,3
        print(i,action, branch_id)
        ## This is for illustration. The AVDC model runs trajectory until failure_step. Then I use expert policy to run. Every time the model changes a branch, I set it as a subgoal.
        if(i==failure_step or (i>failure_step and branch_id !=branch_cache)):
        # if(i>=failure_step):
            # gap images definition can be seen in the next branch. I only want to add gap images when the model can indeed gets to the next subgoal, otherwise I don't think those data useful.
            images+=gap_images
            gap_images=[]
            images.append(image)
            gap=0
        # # If a subgoal takes to long to finish, I save it every 15 steps.
        elif (i>failure_step and gap>15):
            gap_images.append(image)
            gap=0
        # # This is given to you so that you know how the previous failure steps behave.
        elif(i<failure_step):
            fails.append(image)
        # branch_cache is used to check whether the branch_id changes from steps to steps.
        branch_cache=branch_id
        # Increment the gap steps
        if(i>failure_step):
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
        print(done)
    images_to_video(images,'./test.mp4',fps=2)
    images_to_video(fails,'./fail.mp4',fps=10)
