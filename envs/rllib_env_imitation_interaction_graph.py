import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(directory))

import copy
import numpy as np
import argparse
import random

import gym
from gym.spaces import Box

from envs import env_humanoid_imitation_interaction_graph as my_env
import env_renderer as er
import render_module as rm

import pickle
import gzip

import rllib_model_torch as policy_model
from collections import deque

from ray.rllib.utils import try_import_torch
torch, nn = try_import_torch()

from fairmotion.core.motion import Pose, Motion
from fairmotion.core.velocity import MotionWithVelocity
from fairmotion.data import bvh
from fairmotion.ops import conversions
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv

def get_bool_from_input(question):
    answer = input('%s [y/n]?:'%question)
    if answer == 'y' or answer == 'yes':
        answer = True
    elif answer == 'n' or answer == 'no':
        answer = False
    else:
        raise Exception('Please enter [y/n]!')
    return answer

def get_int_from_input(question):
    answer = input('%s [int]?:'%question)
    try:
       answer = int(answer)
    except ValueError:
       print("That's not an integer!")
       return
    return answer

def get_float_from_input(question):
    answer = input('%s [float]?:'%question)
    try:
       answer = float(answer)
    except ValueError:
       print("That's not a float number!")
       return
    return answer

class HumanoidImitationInteractionGraph(TaskSettableEnv):
    def __init__(self, env_config):
        self.base_env = my_env.Env(env_config)
        assert self.base_env._num_agent == 1
        
        ob_scale = 1000.0
        dim_state = self.base_env.dim_state(0)

        dim_state_task = self.base_env.dim_state_task(0)
        dim_state_body = dim_state-dim_state_task



        dim_action = self.base_env.dim_action(0)
        action_range_min, action_range_max = self.base_env.action_range(0)

        self.observation_space = \
            Box(-ob_scale * np.ones(dim_state),
                ob_scale * np.ones(dim_state),
                dtype=np.float64)
        self.observation_space_body = \
            Box(-ob_scale * np.ones(dim_state_body),
                ob_scale * np.ones(dim_state_body),
                dtype=np.float64)
        self.observation_space_task = \
            Box(-ob_scale * np.ones(dim_state_task),
                ob_scale * np.ones(dim_state_task),
                dtype=np.float64)
        self.dim_interaction = self.base_env.dim_interaction(0)
        self.dim_feature = self.base_env.dim_feature(0)
        self.num_state_interaction = self.base_env.num_interaction(0)
        self.sparse_rep = self.base_env.is_sparse_interaction()
        action_range_min, action_range_max = self.base_env.action_range(0)

        self.observation_space = \
            Box(-ob_scale * np.ones(dim_state),
                ob_scale * np.ones(dim_state),
                dtype=np.float64)

        self.action_space = \
            Box(action_range_min,
                action_range_max,
                dtype=np.float64)

    def state(self):
        return self.base_env.state(idx=0)

    def reset(self, info={}):
        if not self.base_env._initialized:
            self.base_env.create()
        self.base_env.reset(info)
        return self.base_env.state(idx=0)
    def format_state(self,state):
        formatted_state = {
            "verts":[],
            "edges":[],
        }
        seg_length = self.dim_interaction
        dim_state = int(np.product(self.observation_space.shape))
        dim_fc = dim_state - self.dim_interaction*self.num_state_interaction
        interaction_point_dim = self.dim_feature[0] * self.dim_feature[1]
         
        
        # seg = obs[:,self._dim_fc+i*seg_length:self._dim_fc+(i+1)*seg_length]

        # interaction_point_dim = self._interaction_feature_dim[0]*self._interaction_feature_dim[1]
        # num_edges = seg[:,interaction_point_dim].to(torch.long)

        # total_num_edges = self._interaction_feature_dim[0]*self._interaction_feature_dim[0]
        # seg_interaction_points = seg[:,:interaction_point_dim]
        # seg_interaction_points = seg_interaction_points.reshape(seg_interaction_points.shape[0],self._interaction_feature_dim[0],self._interaction_feature_dim[1])

        # seg_interaction_edges_connectivity = seg[:,interaction_point_dim+1:interaction_point_dim+1+total_num_edges*2]
        # seg_interaction_edges_connectivity = seg_interaction_edges_connectivity.reshape(seg_interaction_edges_connectivity.shape[0],2,total_num_edges)
        # seg_interaction_edges_connectivity = seg_interaction_edges_connectivity[:,:,:max_edges]
        # seg_interaction_edges_connectivity = seg_interaction_edges_connectivity.to(torch.long)
        
        # seg_interaction_edges_features = seg[:,interaction_point_dim+1+total_num_edges*2:]
        # seg_interaction_edges_features = seg_interaction_edges_features.reshape(seg_interaction_edges_features.shape[0],-1,self._interaction_feature_dim[1])
        # seg_interaction_edges_features = seg_interaction_edges_features[:,:max_edges,:]
        
        # seg_length = self._interaction_obs_dim
        # seg = obs[:,self._dim_fc+i*seg_length:self._dim_fc+(i+1)*seg_length]
        # interaction_point_dim = self._interaction_feature_dim[0]*self._interaction_feature_dim[1]
        # seg = seg[:, interaction_point_dim:]
        # seg = seg.reshape(seg.shape[0],self._interaction_feature_dim[1],self._interaction_feature_dim[0],self._interaction_feature_dim[0])

        
        for i in range(self.num_state_interaction):
            
            seg = state[dim_fc+i*seg_length:dim_fc+(i+1)*seg_length]
            if self.sparse_rep:
                seg_interaction_points = seg[:interaction_point_dim]
                seg_interaction_points.reshape(self.dim_feature[0],self.dim_feature[1])

                pass
            else:
                seg_interaction_points = seg[:interaction_point_dim]
                seg_interaction_points.reshape(self.dim_feature[0],self.dim_feature[1])
                
                seg_edges = seg[interaction_point_dim:]
                seg_edges = seg_edges.reshape(self.dim_feature[0],self.dim_feature[0],self.dim_feature[1])
                seg_edges = np.moveaxis(seg_edges,-1,0)

                formatted_state['verts'].append(seg_interaction_points)
                formatted_state['edges'].append(seg_edges)


        return formatted_state
    def step(self, action):
        
        # cnt = 0
        
        # if self.base_env._use_base_residual_linear_force:
        #     base_residual_linear_force = action[cnt:cnt+3]
        #     cnt += 3
        # else:
        #     base_residual_linear_force = None
        
        # if self.base_env._use_base_residual_angular_force:
        #     base_residual_angular_force = action[cnt:cnt+3]
        #     cnt += 3
        # else:
        #     base_residual_angular_force = None

        # target_pose = action[cnt:]
        
        # action_dict = {
        #     'base_residual_linear_force': base_residual_linear_force,
        #     'base_residual_angular_force': base_residual_angular_force,
        #     'target_pose': target_pose,
        # }

        rew, info = self.base_env.step([action])
        obs = self.state()
        eoe = self.base_env._end_of_episode
        if self.base_env._verbose:
            print(rew)
        return obs, rew[0], eoe, info[0]

class EnvRenderer(er.EnvRenderer):
    def __init__(self, trainers, config, **kwargs):
        from fairmotion.viz.utils import TimeChecker
        # kwargs['renderer'] = 'bullet_native'
        super().__init__(**kwargs)
        self.trainer = trainers[0]
        self.config = config
        self.time_checker_auto_play = TimeChecker()
        self.explore = False
        self.bgcolor=[1.0, 1.0, 1.0, 1.0]
        self.latent_random_sample = 0
        self.latent_random_sample_methods = [None, "gaussian", "uniform", "softmax", "hardmax"]
        self.cam_params = deque(maxlen=30)
        self.cam_param_offset = None
        self.replay = False
        self.replay_cnt = 0
        self.replay_data = {}
        self.replay_render_interval = 15
        self.replay_render_alpha = 0.5
        self.reset()
    def use_default_ground(self):
        return True
    def get_v_up_env_str(self):
        return self.env.base_env._v_up_str
    def get_ground(self):
        return self.env.base_env._ground
    def get_pb_client(self):
        return self.env.base_env._pb_client
    def reset(self, info={}):
        self.data = {
            'reward_info':{},
            'reward': deque(maxlen=300),
            'time': deque(maxlen=300),
            'states': deque(maxlen=300),
            'custom': {}
        }
        self.replay_cnt = 0
        if self.replay:
            self.set_pose()
        else:
            self.env.reset(info)
            self.policy_hidden_state = self.trainer.get_policy().get_initial_state()
            motion = Motion(
                skel=self.env.base_env._base_motion[0].skel,
                fps=self.env.base_env._base_motion[0].fps,
            )
            self.replay_data = {
                'motion': motion,
                'joint_data': list(),
                'link_data': list(),
                'others': {},
            }
        self.cam_params.clear()
        param = self._get_cam_parameters()
        for i in range(self.cam_params.maxlen):
            self.cam_params.append(param)
    def collect_replay_data(self,idx):
        env = self.env.base_env
        sim_agent = env._sim_agent[idx]
        motion = self.replay_data['motion']
        motion.add_one_frame(
            sim_agent.get_pose(motion.skel, apply_height_offset=False).data)
        joint_data, link_data = env.get_render_data(0)
        self.replay_data['joint_data'].append(joint_data)
        self.replay_data['link_data'].append(link_data)
    def set_pose(self):
        if self.replay_data['motion'].num_frames() == 0: 
            return
        motion = self.replay_data['motion']
        pose = motion.get_pose_by_frame(self.replay_cnt)
        self.env.base_env._sim_agent[0].set_pose(pose)
    def one_step(self, explore=None, collect_data=False):
        self.cam_params.append(self._get_cam_parameters())
        
        if self.replay:
            self.set_pose()
            self.replay_cnt = \
                min(self.replay_data['motion'].num_frames()-1, self.replay_cnt+1)
            return
        
        if explore is None:
            explore = self.explore
        
        import time
        start_time = time.time()
        # Run the environment
        s1 = self.env.state()
        print("---State Execution: %.4f seconds ---" % (time.time() - start_time))

        policy = self.trainer.get_policy()
        action = self.trainer.compute_single_action(
                    s1, 
                    explore=explore)
        # Step forward
        s2, rew, eoe, info = self.env.step(action)
        print("---Step Execution: %.4f seconds ---" % (time.time() - start_time))
        self.data['reward_info']=info['rew_info']
        self.data['reward'].append(rew)
        self.data['time'].append(self.get_elapsed_time())
        self.data['states'].append((self.env.format_state(s1),self.env.format_state(s2)))
        # self.get_custom_data()
        for i in range(2):
            self.collect_replay_data(i)
        
        return s2, rew, eoe, info
    def get_custom_data(self):
        base_env = self.env.base_env
        sim_interaction_points = base_env.get_all_interaction_points('sim',0)
        ref_interaction_points = base_env.get_all_interaction_points('kin',0)

        lknee_idx = base_env._interaction_joints.index(base_env._sim_agent[0]._char_info.joint_idx['lknee'])
        lfoot_idx = base_env._interaction_joints.index(base_env._sim_agent[0]._char_info.joint_idx['lankle'])

        rknee_idx = base_env._interaction_joints.index(base_env._sim_agent[0]._char_info.joint_idx['rknee'])
        rfoot_idx = base_env._interaction_joints.index(base_env._sim_agent[0]._char_info.joint_idx['rankle'])

        lelbow_idx = base_env._interaction_joints.index(base_env._sim_agent[0]._char_info.joint_idx['lelbow'])
        lhand_idx = base_env._interaction_joints.index(base_env._sim_agent[0]._char_info.joint_idx['lwrist'])

        relbow_idx = base_env._interaction_joints.index(base_env._sim_agent[0]._char_info.joint_idx['relbow'])
        rhand_idx = base_env._interaction_joints.index(base_env._sim_agent[0]._char_info.joint_idx['rwrist'])

        sim_lknee_pos = sim_interaction_points[lknee_idx]
        sim_lfoot_pos = sim_interaction_points[lfoot_idx]
        sim_rknee_pos = sim_interaction_points[rknee_idx]
        sim_rfoot_pos = sim_interaction_points[rfoot_idx]
        sim_lelbow_pos = sim_interaction_points[lelbow_idx]
        sim_lhand_pos = sim_interaction_points[lhand_idx]
        sim_relbow_pos = sim_interaction_points[relbow_idx]
        sim_rhand_pos = sim_interaction_points[rhand_idx]

        kin_lknee_pos = ref_interaction_points[lknee_idx]
        kin_lfoot_pos = ref_interaction_points[lfoot_idx]
        kin_rknee_pos = ref_interaction_points[rknee_idx]
        kin_rfoot_pos = ref_interaction_points[rfoot_idx]
        kin_lelbow_pos = ref_interaction_points[lelbow_idx]
        kin_lhand_pos = ref_interaction_points[lhand_idx]
        kin_relbow_pos = ref_interaction_points[relbow_idx]
        kin_rhand_pos = ref_interaction_points[rhand_idx]   
        
        self.data['custom']['key_joints'] = {
            "sim_lknee":sim_lknee_pos,
            "sim_lfoot":sim_lfoot_pos,
            "sim_rknee":sim_rknee_pos,
            "sim_rfoot":sim_rfoot_pos,
            "sim_lelbow":sim_lelbow_pos,
            "sim_lwrist":sim_lhand_pos,
            "sim_relbow":sim_relbow_pos,
            "sim_rwrist":sim_rhand_pos,
            
            "kin_lknee":kin_lknee_pos,
            "kin_lfoot":kin_lfoot_pos,
            "kin_rknee":kin_rknee_pos,
            "kin_rfoot":kin_rfoot_pos,
            "kin_lelbow":kin_lelbow_pos,
            "kin_lwrist":kin_lhand_pos,
            "kin_relbow":kin_relbow_pos,
            "kin_rwrist":kin_rhand_pos,
        }
    def extra_render_callback(self):
        model = self.trainer.get_policy().model

        if self.rm.get_flag('custom4'):
            import matplotlib.pyplot as plt
            # s1,s2 = self.data['states'][-1]

            fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(7, 7))
            sim_inter_mesh = self.env.base_env.compute_interaction_mesh(self.env.base_env._sim_interaction_points[0])
            kin_inter_mesh = self.env.base_env.compute_interaction_mesh(self.env.base_env._kin_interaction_points[0])

            sim_inter_mesh_p_dist = np.linalg.norm(sim_inter_mesh[:,:,:3],axis=2)
            sim_inter_mesh_v_dist = np.linalg.norm(sim_inter_mesh[:,:,3:],axis=2)


            kin_inter_mesh_p_dist = np.linalg.norm(kin_inter_mesh[:,:,:3],axis=2)
            kin_inter_mesh_v_dist = np.linalg.norm(kin_inter_mesh[:,:,3:],axis=2)

            axes[0,0].matshow(sim_inter_mesh_p_dist,cmap='seismic')
            for (i,j),z in np.ndenumerate(sim_inter_mesh_p_dist):
                axes[0,0].text(j,i,'%.2f'%(z),ha='center',va='center',fontsize="xx-small")
            axes[0,0].set_title("Sim Pos Dist")

            axes[0,1].matshow(sim_inter_mesh_v_dist,cmap='seismic')
            for (i,j),z in np.ndenumerate(sim_inter_mesh_v_dist):
                axes[0,1].text(j,i,'%.2f'%(z),ha='center',va='center',fontsize="xx-small")
            axes[0,1].set_title("Sim Vel Dist")

            axes[1,0].matshow(kin_inter_mesh_p_dist,cmap='seismic')
            for (i,j),z in np.ndenumerate(kin_inter_mesh_p_dist):
                axes[1,0].text(j,i,'%.2f'%(z),ha='center',va='center',fontsize="xx-small")
            axes[1,0].set_title("Ref Pos Dist")

            axes[1,1].matshow(kin_inter_mesh_v_dist,cmap='seismic')
            for (i,j),z in np.ndenumerate(kin_inter_mesh_v_dist):
                axes[1,1].text(j,i,'%.2f'%(z),ha='center',va='center',fontsize="xx-small")   
            axes[1,1].set_title("Ref Vel Dist")
        
            fig.tight_layout()
            plt.show()
            self.rm.flag['custom4'] = False
        self.env.base_env.render(self.rm)
    def extra_overlay_callback(self):
        w, h = self.window_size

        w_curr = 50
        h_curr = 50
        if len(self.data['reward'])>0:

            reward = self.data['reward']
            time = np.arange(len(reward))/reward.maxlen
            x_range = (0, 1.0)
            y_range=(0,1)
            w_size=150
            h_size=150
            pad_len=10
            origin = (np.array([w_curr, h_size+h_curr])).flatten()
            self.rm.gl_render.render_graph_base_2D(origin,w_size,pad_len)
            self.rm.gl_render.render_graph_data_point_2D(
                x_data=[time[-1]],
                y_data=[reward[-1]],
                x_range=x_range,
                y_range=y_range,
                color=[1.0, 0.0, 0.0, 1.0],
                point_size=5.0,
                origin=origin,
                axis_len=w_size,
                pad_len=pad_len,
            )

            self.rm.gl_render.render_graph_data_line_2D(
                    x_data=time,
                    y_data=reward,
                    x_range=x_range,
                    y_range=y_range,
                    color=[1.0, 0.0, 0.0,0.5],
                    line_width=2.0,
                    origin=origin,
                    axis_len=w_size,
                    pad_len=pad_len,
            )
            font = self.rm.glut.GLUT_BITMAP_9_BY_15

            self.rm.gl_render.render_text(
                "Reward: %.4f"%(reward[-1]), pos=[w_curr,h_curr], font=font)

            h_curr +=150


        env = self.env.base_env
        font = self.rm.glut.GLUT_BITMAP_9_BY_15
        ref_motion_name = env._ref_motion_file_names[0][env._ref_motion_idx[0]]

        self.rm.gl_render.render_text(
            "File: %s"%ref_motion_name, pos=[0.05*w, 0.05*h], font=font)
        
        if self.rm.flag['overlay_text']:
            total_rwd_info = self.data['reward_info']
            h_curr = 0.5*h
            if total_rwd_info:
                children_rwd_info = total_rwd_info['child_nodes']
                
                all_rwd_info = [total_rwd_info] + children_rwd_info
                for i,rwd_info in enumerate(all_rwd_info):
                    name = rwd_info['name']
                    val = rwd_info['value']
                    self.rm.gl_render.render_text("%s: %.4f"%(name,val), pos=[0.05*w , 0.5*h + i*0.02*h], font=font)
                    h_curr += 0.02*h

            if self.data['custom'].get('key_joints'):
                key_joints = self.data['custom'].get('key_joints')

                sim_rknee = key_joints['sim_rknee']
                sim_lelbow = key_joints['sim_lelbow']
                kin_rknee = key_joints['kin_rknee']
                kin_lelbow = key_joints['kin_lelbow']   
                sim_rknee_lelbow_p_dist = np.linalg.norm(sim_rknee[:3]-sim_lelbow[:3])
                sim_rknee_lelbow_v_dist = np.linalg.norm(sim_rknee[3:]-sim_lelbow[3:])
                kin_rknee_lelbow_p_dist = np.linalg.norm(kin_rknee[:3]-kin_lelbow[:3])
                kin_rknee_lelbow_v_dist = np.linalg.norm(kin_rknee[3:]-kin_lelbow[3:])
                self.rm.gl_render.render_text("Sim lelbow/rknee pos dist: %.4f, vel dist: %.4f"%(sim_rknee_lelbow_p_dist,sim_rknee_lelbow_v_dist), pos=[0.05*w , h_curr+0.02*h], font=font)
                h_curr += 0.02*h
                self.rm.gl_render.render_text("Kin lelbow/rknee pos dist: %.4f, vel dist: %.4f"%(kin_rknee_lelbow_p_dist,kin_rknee_lelbow_v_dist), pos=[0.05*w , h_curr+0.02*h], font=font)
                h_curr += 0.02*h
                
                sim_lknee = key_joints['sim_lknee']
                sim_relbow = key_joints['sim_relbow']
                kin_lknee = key_joints['kin_lknee']
                kin_relbow = key_joints['kin_relbow']
                sim_lknee_relbow_p_dist = np.linalg.norm(sim_lknee[:3]-sim_relbow[:3])
                sim_lknee_relbow_v_dist = np.linalg.norm(sim_lknee[3:]-sim_relbow[3:])
                kin_lknee_relbow_p_dist = np.linalg.norm(kin_lknee[:3]-kin_relbow[:3])
                kin_lknee_relbow_v_dist = np.linalg.norm(kin_lknee[3:]-kin_relbow[3:])
                self.rm.gl_render.render_text("Sim relbow/lknee pos dist: %.4f, vel dist: %.4f"%(sim_lknee_relbow_p_dist,sim_lknee_relbow_v_dist), pos=[0.05*w , h_curr+0.02*h], font=font)
                h_curr += 0.02*h
                self.rm.gl_render.render_text("Kin relbow/lknee pos dist: %.4f, vel dist: %.4f"%(kin_lknee_relbow_p_dist,kin_lknee_relbow_v_dist), pos=[0.05*w , h_curr+0.02*h], font=font)
                h_curr += 0.02*h


                sim_lwrist = key_joints['sim_lwrist']
                sim_rfoot = key_joints['sim_rfoot']
                kin_lwrist = key_joints['kin_lwrist']
                kin_rfoot = key_joints['kin_rfoot']   
                sim_lwrist_rfoot_p_dist = np.linalg.norm(sim_lwrist[:3]-sim_rfoot[:3])
                sim_lwrist_rfoot_v_dist = np.linalg.norm(sim_lwrist[3:]-sim_rfoot[3:])
                kin_lwrist_rfoot_p_dist = np.linalg.norm(kin_lwrist[:3]-kin_rfoot[:3])
                kin_lwrist_rfoot_v_dist = np.linalg.norm(kin_lwrist[3:]-kin_rfoot[3:])
                self.rm.gl_render.render_text("Sim lwrist/rfoot pos dist: %.4f, vel dist: %.4f"%(sim_lwrist_rfoot_p_dist,sim_lwrist_rfoot_v_dist), pos=[0.05*w , h_curr+0.02*h], font=font)
                h_curr += 0.02*h
                self.rm.gl_render.render_text("Kin lwrist/rfoot pos dist: %.4f, vel dist: %.4f"%(kin_lwrist_rfoot_p_dist,kin_lwrist_rfoot_v_dist), pos=[0.05*w , h_curr+0.02*h], font=font)
                h_curr += 0.02*h

                sim_rwrist = key_joints['sim_rwrist']
                sim_lfoot = key_joints['sim_lfoot']
                kin_rwrist = key_joints['kin_rwrist']
                kin_lfoot = key_joints['kin_lfoot']   
                sim_rwrist_lfoot_p_dist = np.linalg.norm(sim_rwrist[:3]-sim_lfoot[:3])
                sim_rwrist_lfoot_v_dist = np.linalg.norm(sim_rwrist[3:]-sim_lfoot[3:])
                kin_rwrist_lfoot_p_dist = np.linalg.norm(kin_rwrist[:3]-kin_lfoot[:3])
                kin_rwrist_lfoot_v_dist = np.linalg.norm(kin_rwrist[3:]-kin_lfoot[3:])
                self.rm.gl_render.render_text("Sim rwrist/lfoot pos dist: %.4f, vel dist: %.4f"%(sim_rwrist_lfoot_p_dist,sim_rwrist_lfoot_v_dist), pos=[0.05*w , h_curr+0.02*h], font=font)
                h_curr += 0.02*h
                self.rm.gl_render.render_text("Kin rwrist/lfoot pos dist: %.4f, vel dist: %.4f"%(kin_rwrist_lfoot_p_dist,kin_rwrist_lfoot_v_dist), pos=[0.05*w , h_curr+0.02*h], font=font)
                h_curr += 0.02*h

    def extra_idle_callback(self):
        time_elapsed = self.time_checker_auto_play.get_time(restart=False)
        if self.rm.flag['auto_play'] and time_elapsed >= self.env.base_env._dt_con:
            self.time_checker_auto_play.begin()
            self.one_step(collect_data=True)
    def extra_keyboard_callback(self, key):
        if key == b'r':
            print("Reset w/o replay")
            self.replay = False
            self.reset({'start_time': np.array([0])})
        elif key == b'R':
            time = get_float_from_input("Enter start time")
            self.reset({'start_time': np.array([time])})
        elif key == b'p':
            print("Reset w/ replay")
            self.replay = False
            self.reset()
        elif key == b']':
            if self.replay:
                self.replay_cnt = \
                    min(self.replay_data[0]['motion'].num_frames()-1, self.replay_cnt+1)
                self.set_pose()
        elif key == b'[':
            if self.replay:
                self.replay_cnt = max(0, self.replay_cnt-1)
                self.set_pose()
        elif key == b'+':
            self.replay_render_interval = \
                min(90, self.replay_render_interval+5)
            print('replay_render_interval', self.replay_render_interval)
        elif key == b'-':
            self.replay_render_interval = \
                max(5, self.replay_render_interval-5)
            print('replay_render_interval', self.replay_render_interval)
        elif key == b'>':
            self.replay_render_alpha = \
                min(1.0, self.replay_render_alpha+0.1)
            print('replay_render_alpha', self.replay_render_alpha)
        elif key == b'<':
            self.replay_render_alpha = \
                max(0.1, self.replay_render_alpha-0.1)
            print('replay_render_alpha', self.replay_render_alpha)
        elif key == b' ':
            self.time_checker_auto_play.begin()
            self.one_step()
        elif key == b'e':
            self.explore = not self.explore
            print('Exploration:', self.explore)
        elif key == b'E':
            model = self.trainer.get_policy().model
            exp_std = get_float_from_input("Exploration Std")
            assert exp_std >= 0.0
            model.set_exploration_std(exp_std)
        elif key == b'w':
            print('Save Model Weights...')
            model = self.trainer.get_policy().model
            if hasattr(model, 'save_weights'):
                model.save_weights('data/temp/policy_weights.pt')
            if hasattr(model, 'save_weights_body_encoder'):
                model.save_weights_body_encoder('data/temp/task_agnostic_policy_body_encoder.pt')
            if hasattr(model, 'save_weights_task_encoder'):
                model.save_weights_task_encoder('data/temp/task_agnostic_policy_task_encoder.pt')
            if hasattr(model, 'save_weights_motor_decoder'):
                model.save_weights_motor_decoder('data/temp/task_agnostic_policy_motor_decoder.pt')
            if hasattr(model, 'save_weights_seperate'):
                model.save_weights_seperate(
                    file_gate='data/temp/imitation_policy_gate',
                    file_experts='data/temp/imitation_policy_expert',
                    file_helpers='data/temp/imitation_policy_helper')
            print('Done.')
        elif key == b"I":
            save_dir = input("Enter directory for saving: ")
            try:
                os.makedirs(save_dir, exist_ok=True)
            except OSError:
                print("Invalid Subdirectory")
                return
            all_iteractions = {}

            for i in range(len(self.env.base_env._ref_motion_all[0])):
                self.reset({
                    'start_time': np.array([0.0]),
                    'ref_motion_id': [i],
                    })
                
                interaction = self.save_interaction_rollout()
                all_iteractions[i] = interaction
            if save_dir:
                save_dir_i = os.path.join(save_dir, "interaction.pkl")
                pickle.dump(all_iteractions,open(save_dir_i,"wb"))
                
        elif key == b's' or key == b'S':
            save_image = get_bool_from_input("Save image")
            save_motion = get_bool_from_input("Save motion")
            save_motion_only_success = False
            if save_motion:
                save_motion_only_success = get_bool_from_input("Save success motion only")
            save_dir = None
            if save_image or save_motion:
                ''' Read a directory for saving images and try to create it '''
                save_dir = input("Enter directory for saving: ")
                try:
                    os.makedirs(save_dir, exist_ok=True)
                except OSError:
                    print("Invalid Subdirectory")
                    return
            if key == b's':
                print('Recording the Current Scene...')
                ''' Read maximum end time '''
                end_time = get_float_from_input("Enter max end-time (sec)")
                ''' Read number of iteration '''
                num_iter = get_int_from_input("Enter num iteration")
                ''' Start each episode at zero '''
                reset_at_fixed_time = get_bool_from_input("Always reset at a fixed time")
                if reset_at_fixed_time:
                    start_time = get_float_from_input("Enter start time")
                ''' Read falldown check '''
                check_falldown = get_bool_from_input("Terminate when falldown")
                ''' Read end_of_motion check '''
                check_end_of_motion = get_bool_from_input("Terminate when reaching the end of motion")
                for i in range(num_iter):
                    if reset_at_fixed_time:
                        self.reset({'start_time': np.array([start_time])})
                    else:
                        self.reset()
                    if save_dir:
                        save_dir_i = os.path.join(save_dir, str(i))
                    else:
                        save_dir_i = None
                    time_elapsed = self.record_a_scene(
                        save_dir=save_dir_i, 
                        save_image=save_image,
                        save_motion=save_motion,
                        save_motion_only_success=save_motion_only_success,
                        end_time=end_time, 
                        check_falldown=check_falldown, 
                        check_end_of_motion=check_end_of_motion)
                print('Done.')
            elif key == b'S':
                print('Recording the Entire Motions...')
                for i in range(len(self.env.base_env._ref_motion_all[0])):
                    self.reset({
                        'start_time': np.array([0.0]),
                        'ref_motion_id': [i],
                        })
                    if save_dir:
                        save_dir_i = os.path.join(save_dir, str(i))
                    else:
                        save_dir_i = None
                    self.record_a_scene(
                        save_dir=save_dir_i,
                        save_image=save_image,
                        save_motion=save_motion)
                print('Done.')
        elif key == b'?':
            filename = os.path.join(save_dir_i, "replay.pkl")
            pickle.dump(self.replay_data, open(filename, "wb"))
            print(filename)
        elif key == b'/':
            print('Load Replay Data...')
            name = input("Enter data file: ")
            with open(name, "rb") as f:
                self.replay_data = pickle.load(f)
            self.replay = True
            self.reset()
            print('Done.')
        elif key == b'g':
            ''' Read maximum end time '''
            end_time = get_float_from_input("Enter max end-time (sec)")
            ''' Read number of iteration '''
            run_every_motion = get_bool_from_input("Run for every ref_motion?")
            ''' Read number of iteration '''
            num_iter = get_int_from_input("Enter num iteration")
            ''' Start each episode at zero '''
            reset_at_zero = get_bool_from_input("Always reset at 0s")
            ''' Verbose '''
            verbose = get_bool_from_input("Verbose")
            ''' Result file '''
            result_file = input("Result file:")
            ''' Save Motion '''
            save_motion = get_bool_from_input("Save motion")
            save_motion_dir = None
            if save_motion:
                save_motion_dir = input("Enter directory for saving: ")
                try:
                    os.makedirs(save_motion_dir, exist_ok=True)
                except OSError:
                    print("Invalid Subdirectory")
                    return

            while True:
                if self.latent_random_sample_methods[self.latent_random_sample] == 'gaussian':
                    break
                self.latent_random_sample = (self.latent_random_sample+1)%len(self.latent_random_sample_methods)

            motor_decoders = [None]
             
            print("----------------------------")
            print("latent_random_sample:", self.latent_random_sample_methods[self.latent_random_sample])
            print("exploration:", self.explore)
            print("----------------------------")

            with open(result_file, 'w') as f:
                f.write("------------------VAE MD Performace Test------------------\n")

                if run_every_motion:
                    ref_motions = np.arange(len(self.env.base_env._ref_motion_all[0]))
                else:
                    ref_motions = [0]

                ''' Use fixed start-time value for comparison '''
                start_times_all = []
                for i in range(len(ref_motions)):
                    if reset_at_zero:
                        start_times = np.zeros(num_iter)
                    else:
                        self.reset({'ref_motion_id': [0]})
                        start_times = np.linspace(
                            start=0.0,
                            stop=self.env.base_env._ref_motion[i].length(),
                            num=num_iter)
                        print("start_times:", start_times)
                    start_times_all.append(start_times)

                f.writelines([md+"\n" for md in motor_decoders])

                for k, md in enumerate(motor_decoders):
                    if md:
                        model = self.trainer.get_policy().model
                        model.load_weights_motor_decoder(md)
                    print("Loaded:", md)
                    time_elapsed = []
                    for i in ref_motions:
                        if verbose:
                            print('ref_motion[%d]'%(i))
                        for j in range(num_iter):
                            save_motion_name = "motion_%d_%d_%d.bvh" % (k, i, j)
                            self.reset({
                                'ref_motion_id':[i], 
                                'start_time': [start_times_all[i][j]]})
                            time_elapsed.append(self.record_a_scene(
                                save_dir=save_motion_dir, 
                                save_image=False,
                                save_motion=save_motion,
                                save_motion_name=save_motion_name,
                                end_time=end_time, 
                                check_falldown=True, 
                                check_end_of_motion=False, 
                                verbose=verbose))
                            # print(time_elapsed)
                    msg = "%4.4f\t%4.4f" % (np.mean(time_elapsed), np.std(time_elapsed))
                    print(msg)
                    f.write("%s\n"%msg)
                    f.flush()
                f.close()
            print("----------------------------")
        elif key == b'l':
            file = input("Enter Checkpoint:")
            if os.path.isfile(file):
                model = self.trainer.get_policy().model
                model.load_weights_motor_decoder(file)
                print(file)
        elif key == b'L':
            model = self.trainer.get_policy().model
            is_task_agnostic_policy = hasattr(model, 'task_encoder_variable')
            print('----------------------------')
            print('Extracting State-Action Pairs')
            print('----------------------------')
            ''' Read a directory for saving images and try to create it '''
            output_dir = input("Enter output dir: ")
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError:
                print("Invalid Subdirectory")
                return
            iter_per_episode = get_int_from_input("Iteration Per Episode")
            window_size = get_float_from_input("Window Size (sec)")
            stride = get_float_from_input("Stride (sec)")
            state_type = input("State Type: ")
            exp_std = get_float_from_input("Exploration Std")
            assert state_type in ["facing", "facing_R6_h", "root_R6_h"]
            assert exp_std >= 0.0
            model.set_exploration_std(exp_std)
            def state_body_custom(type):
                return self.env.base_env.state_body(
                    idx=0, 
                    agent="sim", 
                    type=type, 
                    return_stacked=True)
            data = {
                'iter_per_episode': iter_per_episode,
                'dim_state': self.env.base_env.dim_state(0),
                'dim_state_body': len(state_body_custom(state_type)),
                'dim_state_task': self.env.base_env.dim_state_task(0),
                'dim_action': self.env.base_env.dim_action(0),
                'episodes': [],
                'exp_std': exp_std,
            }
            base_env = self.env.base_env
            for i in range(len(self.env.base_env._ref_motion_all[0])):
                for j in range(iter_per_episode):
                    cnt_per_trial = 0
                    time_start = -window_size + stride
                    while True:
                        print("\rtime_start: ", time_start, end='')
                        episode = {
                            'time': [],
                            'state': [],
                            'action': [],
                            'action_gt': [],
                            'reward': [],
                            'state_body': [],
                            'state_task': [],
                        }
                        if is_task_agnostic_policy:
                            episode.update({
                                'z_body': [],
                                'z_task': [],
                            })
                        self.env.reset({
                            'ref_motion_id': [i], 
                            'start_time': np.array([max(0.0, time_start)])}
                        )
                        time_elapsed = max(0.0, time_start) - time_start
                        falldown = False
                        cnt_per_window = 0
                        while True:
                            s1 = self.env.state()
                            s1_body = state_body_custom(state_type)
                            s1_task = base_env.state_task(0)
                            a = self.trainer.compute_single_action(s1, explore=True)
                            a_gt = self.trainer.compute_single_action(s1, explore=False)
                            s2, rew, eoe, info = self.env.step(a)
                            t = base_env.get_current_time()
                            time_elapsed += base_env._dt_con
                            episode['time'].append(t)
                            episode['state'].append(s1)
                            episode['action'].append(a)
                            episode['action_gt'].append(a_gt)
                            episode['reward'].append(rew)
                            episode['state_body'].append(s1_body)
                            episode['state_task'].append(s1_task)
                            if is_task_agnostic_policy:
                                z_body = model.body_encoder_variable()[0].detach().numpy()
                                z_task = model.task_encoder_variable()[0].detach().numpy()
                                episode['z_body'].append(z_body)
                                episode['z_task'].append(z_task)
                            cnt_per_window += 1
                            ''' 
                            The policy output might not be ideal 
                            when no future reference motion exists.
                            '''
                            if base_env._ref_motion[0].length()-base_env.get_current_time() \
                               <= base_env._imit_window[-1]:
                                break
                            if time_elapsed >= window_size:
                                break
                            if self.env.base_env._end_of_episode:
                                falldown = True
                                break
                        ''' Include only successful (not falling) episodes '''
                        if not falldown:
                            data['episodes'].append(episode)
                            time_start += stride
                            cnt_per_trial += cnt_per_window
                            ''' End if not enough frames remain '''
                            if base_env._ref_motion[0].length() < time_start + stride:
                                break
                        else:
                            print("\r******FALLDOWN****** Retrying...")
                    print('\n%d pairs were created in %d-th trial of episode%d'%(cnt_per_trial, j, i))
            output_file = os.path.join(
                output_dir, 
                "data_iter=%d,winsize=%.2f,stride=%.2f,state_type=%s,exp_std=%.2f.pkl"%(iter_per_episode, window_size, stride, state_type, exp_std))
            with open(output_file, "wb") as file:
                pickle.dump(data, file)
                print("Saved:", file)
        elif key == b'j':
            print('Save Current Render Data...')
            posfix = input("Enter prefix for the file name:")
            name_joint_data = os.path.join(
                "data/temp", "joint_data_" + posfix +".pkl")
            name_link_data = os.path.join(
                "data/temp", "link_data_" + posfix +".pkl")
            pickle.dump(
                self.data['joint_data'], 
                open(name_joint_data, "wb"))
            pickle.dump(
                self.data['link_data'], 
                open(name_link_data, "wb"))
            print('Done.')
        elif key==b'5':
            if self.rm.flag['kin_model']:
                for i in range(self.env.base_env._num_agent):
                    self.env.base_env._kin_agent[i].change_visual_color([0, 0, 0.5, 1])
            else:
                for i in range(self.env.base_env._num_agent):
                    self.env.base_env._kin_agent[i].change_visual_color([0, 0, 0.5, 0])
        elif key == b'q':
            self.latent_random_sample = (self.latent_random_sample+1)%len(self.latent_random_sample_methods)
            print("latent_random_sample:", self.latent_random_sample_methods[self.latent_random_sample])
        elif key == b'x':
            model = self.trainer.get_policy().model
            exp_std = get_float_from_input("Exploration Std")
            assert exp_std >= 0.0
            model.set_exploration_std(exp_std)
        elif key == b'c':
            agent = self.env.base_env._sim_agent[0]
            h = self.env.base_env.get_ground_height(0)
            d_face, p_face = agent.get_facing_direction_position(h)
            origin = p_face + agent._char_info.v_up_env
            pos = p_face + 4 * (agent._char_info.v_up_env - d_face)
            R_face, _ = conversions.T2Rp(agent.get_facing_transform(h))
            R_face_inv = R_face.transpose()
            origin_offset = np.dot(R_face_inv, self.cam_cur.origin - origin)
            pos_offset = np.dot(R_face_inv, self.cam_cur.pos - pos)
            self.cam_param_offset = (origin_offset, pos_offset)
            print("Set camera offset:", self.cam_param_offset)
        elif key == b'C':
            self.cam_param_offset = None
            print("Clear camera offset")
    # def get_cam_parameters(self, use_buffer=True):
    #     v_up_env_str = self.get_v_up_env_str()
    #     param = {
    #         "translate": {
    #             "target_pos": self.env.base_env._sim_agent[0].get_root_position(),
    #             "ignore_y": v_up_env_str=="y",
    #             "ignore_z": v_up_env_str=="z",
    #         },
    #     }
    #     return param
    def _get_cam_parameters(self, apply_offset=True):
        param = {
            "origin": None, 
            "pos": None, 
            "dist": None,
            "translate": None,
        }
        
        agent = self.env.base_env._sim_agent[0]
        h = self.env.base_env.get_ground_height(0)
        d_face, p_face = agent.get_facing_direction_position(h)
        origin = p_face + agent._char_info.v_up_env

        if self.rm.get_flag("follow_cam") == "pos+rot":
            pos = p_face + 2 * (agent._char_info.v_up_env - d_face)
        else:
            pos = self.cam_cur.pos + (origin - self.cam_cur.origin)
        
        if apply_offset and self.cam_param_offset is not None:
            if self.rm.get_flag("follow_cam") == "pos+rot":
                R_face, _ = conversions.T2Rp(agent.get_facing_transform(h))
                pos += np.dot(R_face, self.cam_param_offset[1])
                origin += np.dot(R_face, self.cam_param_offset[0])
        
        param["origin"] = origin
        param["pos"] = pos
        
        return param
    def get_cam_parameters(self, use_buffer=True):
        if use_buffer:
            param = {
                "origin": np.mean([p["origin"] for p in self.cam_params], axis=0), 
                "pos": np.mean([p["pos"] for p in self.cam_params], axis=0), 
            }
        else:
            param = self._get_cam_parameters()
        return param
    def get_elapsed_time(self):
        return self.env.base_env.get_elapsed_time()
    def save_interaction_rollout(self,):

        self.update_cam()
        time_elapsed = 0
        interactions = []
        inter = copy.deepcopy(self.env.base_env.current_interaction)
        interactions.append(inter)
        while True:
            self.one_step()
            self.draw_GL()
            inter = copy.deepcopy(self.env.base_env.current_interaction)
            interactions.append(inter)
            time_elapsed += self.env.base_env._dt_con

            if self.env.base_env.check_end_of_motion(0):
                break

        return interactions
    def record_a_scene(
        self,
        save_dir, 
        save_image,
        save_motion,
        save_motion_name="motion.bvh",
        save_motion_only_success=False,
        save_replay_name="replay.pkl",
        end_time=None, 
        check_falldown=True, 
        check_end_of_motion=True,
        verbose=True):
        if save_image or save_motion:
            assert save_dir is not None
            try:
                os.makedirs(save_dir, exist_ok = True)
            except OSError:
                print("Invalid Subdirectory")
                return
        if end_time is None or end_time <= 0.0:
            assert check_falldown or check_end_of_motion
        self.update_cam()
        cnt_screenshot = 0
        time_elapsed = 0
        if save_motion:
            motion = copy.deepcopy(self.env.base_env._base_motion[0])
            motion.clear()
        while True:
            self.one_step()
            if save_motion:
                motion.add_one_frame(
                    self.env.base_env._sim_agent[0].get_pose_data(motion.skel))
            if save_image:
                name = 'screenshot_%04d'%(cnt_screenshot)
                self.save_screen(dir=save_dir, name=name, render=True)
                if verbose:
                    print('\rsave_screen(%4.4f) / %s' % \
                        (time_elapsed, os.path.join(save_dir,name)), end=" ")
                cnt_screenshot += 1
            else:
                if verbose:
                    print('\r%4.4f' % (time_elapsed), end=" ")
            time_elapsed += self.env.base_env._dt_con
            agent_name = self.env.base_env._sim_agent[0].get_name()
            if check_falldown:
                if self.env.base_env.check_falldown(0):
                    break
            if check_end_of_motion:
                if self.env.base_env.check_end_of_motion(0):
                    break
            if end_time and time_elapsed >= end_time:
                break
        if save_motion:
            save = True
            if end_time and save_motion_only_success:
                save = time_elapsed >= end_time
            if save:
                bvh.save(
                    motion, 
                    os.path.join(save_dir, save_motion_name),
                    scale=1.0, rot_order="XYZ", verbose=False)
                filename = os.path.join(save_dir, save_replay_name)
                pickle.dump(self.replay_data, open(filename, "wb"))
        if verbose:
            print(" ")
        return time_elapsed

def default_cam(env):
    agent = env.base_env._sim_agent[0]
    # R, p = conversions.T2Rp(agent.get_facing_transform(0.0))    
    v_up_env = agent._char_info.v_up_env
    v_up = agent._char_info.v_up
    v_face = agent._char_info.v_face
    origin = np.zeros(3)
    return rm.camera.Camera(
        pos=3*(v_up+v_face),
        origin=origin, 
        vup=v_up_env, 
        fov=60.0)

env_cls = HumanoidImitationInteractionGraph

def config_override(spec):
    env = env_cls(spec["config"]["env_config"])

    model_config = copy.deepcopy(spec["config"]["model"])
    model = model_config.get("custom_model")
    if model and model == "task_agnostic_policy_type1":
        model_config.get("custom_model_config").update({
            "observation_space_body": copy.deepcopy(env.observation_space_body),
            "observation_space_task": copy.deepcopy(env.observation_space_task),
        })

        # self.dim_interaction = self.base_env.dim_interaction(0)
        # self.dim_feature = self.base_env.dim_feature(0)
        # self.num_state_interaction = self.base_env.num_interaction()
    if model and model == "interaction_net":
        net_type = model_config.get("custom_model_config").get("interaction_net_type")
        print(net_type)
        # if net_type=="conv":
        #     model_config.get("custom_model_config").update({
        #         "interaction_obs_dim": env.dim_state_interaction_full,
        #         "interaction_obs_num": env.num_state_interaction,
        #     })
        # elif net_type == "gcn":
        model_config.get("custom_model_config").update({
            "interaction_obs_dim": env.dim_interaction,
            "interaction_obs_num": env.num_state_interaction,
            "interaction_feature_dim": env.dim_feature,
            "sparse_interaction" : env.sparse_rep
        })
    del env

    config = {
        # "callbacks": {},
        "model": model_config,
    }
    return config
