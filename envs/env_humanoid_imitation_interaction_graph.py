from email.mime import base
import os
from pickle import NONE
import sys
import pickle
from scipy.sparse import coo_matrix

import yaml

from render_module import COLORS_FOR_AGENTS

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(directory))

import copy
import numpy as np

from fairmotion.utils import utils,constants
from fairmotion.ops import conversions, math
from fairmotion.ops import quaternion
from misc.interaction import Interaction
import motion_utils
import time as tt
from envs import env_humanoid_base

class Env(env_humanoid_base.Env):
    def __init__(self, config):
        super().__init__(config)
        
        self._initialized = False
        self._config = copy.deepcopy(config)
        self._ref_motion = self._base_motion
        self._sensor_lookahead = self._config['state'].get('sensor_lookahead', [0.05, 0.15])
        self._start_time = np.zeros(self._num_agent)
        self._repeat_ref_motion = self._config.get('repeat_ref_motion', False)
        self._interaction_choices = self._config['interaction'].get('interaction_choices',['self'])
        self._compute_interaction_connectivity = self._config['interaction'].get('interaction_enabled',False)
        self._is_prune_edges = self._config['interaction'].get('prune_edges',False)
        self._load_ref_interaction = self._config['interaction'].get('load_ref_interaction',None)
        self._interaction_type = self._config['interaction'].get('interaction_type',"interaction_mesh")
        self._interaction_joints = self._config['interaction'].get("interaction_joint_candidates",None)
        self._oppo_interaction_joints = self._config['interaction'].get("oppo_interaction_joint_candidates",None)
        self._object_interaction_joints = self._config['interaction'].get("object_interaction_joint_candidates",None)
        self._interaction_kernel = self._config['interaction'].get("interaction_weight_kernel",0)
        self._interaction_weight_type = self._config['interaction'].get("interaction_weight_type","kin")
        self._sparse_interaction = self._config['interaction'].get("sparse",False)
        self._target_constraints = self._config['interaction'].get('constraints',[])
        self._init_state_dist_type = self._config.get('RSI',{}).get('dist_type','uniform')
        self._init_state_dist_reject = self._config.get('RSI',{}).get('reject',[])
        self._init_dist_bin = None
        self._index_type = self._config['interaction'].get('index_type','joint')
        self._include_object = self._config.get("object",False)
        self.interaction_point_names = self.get_all_interaction_points_name()

        if self._include_object:
            self.load_object()
        self._interaction_joints_xform = []
        self._oppo_interaction_joints_xform = []
        self._object_interaction_joints_xform = []
        self._self_interaction_vert_cnt = 0
        self._oppo_interaction_vert_cnt = 0
        self._object_interaction_vert_cnt = 0
        self._full_matrix_dist = []
        self._full_matrix_dist_raw = []

        self._interaction_filters = {}

        if self._interaction_joints is not None:
            joints = []
            joint_idx_list = self._sim_agent[0]._char_info.joint_idx
            for i in self._interaction_joints:
                if type(i) is dict:
                    joint_name = list(i.keys())[0]
                    translation = np.array(i[joint_name])
                    xform = conversions.p2T(translation)
                else:
                    joint_name = i
                    xform = conversions.p2T([0,0,0])
                joints.append(joint_idx_list[joint_name])
                self._interaction_joints_xform.append(xform)
                
            self._interaction_joints = joints
            self._self_interaction_vert_cnt = len(self._interaction_joints)

        if self._oppo_interaction_joints is not None:
            joints = []
            joint_idx_list = self._sim_agent[0]._char_info.joint_idx
            for i in self._oppo_interaction_joints:
                if type(i) is dict:
                    joint_name = list(i.keys())[0]
                    translation = np.array(i[joint_name])
                    xform = conversions.p2T(translation)
                else:
                    joint_name = i
                    xform = conversions.p2T([0,0,0])
                joints.append(joint_idx_list[joint_name])
                self._oppo_interaction_joints_xform.append(xform)
            self._oppo_interaction_joints = joints
            self._oppo_interaction_vert_cnt = len(self._oppo_interaction_joints)
            # joints = []
            # joint_idx_list = self._sim_agent[0]._char_info.joint_idx
            # [joints.append(joint_idx_list[i]) for i in self._oppo_interaction_joints]

            # self._oppo_interaction_joints = joints   
        if self._object_interaction_joints:
            ''' Include interaction vertices for object'''
            joints = []
            joint_idx_list = self._obj_sim_agent[0]._char_info.joint_idx
            for i in self._object_interaction_joints:
                if type(i) is dict:
                    joint_name = list(i.keys())[0]
                    translation = np.array(i[joint_name])
                    xform = conversions.p2T(translation)
                else:
                    joint_name = i
                    xform = conversions.p2T([0,0,0])
                joints.append(joint_idx_list[joint_name])
                self._object_interaction_joints_xform.append(xform)
            self._object_interaction_joints = joints
            self._object_interaction_vert_cnt = len(self._object_interaction_joints)
            # joints = []
            # joint_idx_list = self._sim_agent[0]._char_info.joint_idx
            # [joints.append(joint_idx_list[i]) for i in self._object_interaction_joints]
            # self._object_interaction_joints = joints   
            pass        


        self._prune_edges = self.remove_neighbour_connectivity()
        self.current_interaction = None

        ''' Add environemtn objects '''
        environment_file = self._config['character'].get('environment_file')
        self.env_obj = None
        if environment_file:
            tf_chair = conversions.R2Q(constants.eye_R())
            self.env_obj = self._pb_client.loadURDF(
                environment_file[0], [0.0, 0.0, 0.0], tf_chair, useFixedBase=True)
        if config.get('lazy_creation'):
            if self._verbose:
                print('The environment was created in a lazy fashion.')
                print('The function \"create\" should be called before it')
            return   
        self.create()
    
    def load_object(self):
        import importlib.util
        import sim_agent
        project_dir        = self._config['project_dir']
        obj_info_module   = self._config['object']["obj_info_module"]
        sim_obj_file      = self._config['object'].get('sim_obj_file')
        ref_motion_file    = self._config['object'].get('ref_motion_file')
        ref_motion_db = self._config['object'].get('ref_motion_db')
        ref_motion_file = motion_utils.collect_motion_files(project_dir, ref_motion_db)

        ref_motion_scale             = \
            self._config["object"].get("ref_motion_scale", [np.ones(len(obj_info_module))])
        self._num_obj = len(sim_obj_file)
        self._obj_info = []
        self._obj_sim_agent = []
        self._obj_kin_agent = []
        self._obj_ref_motion_all = []
        self._obj_ref_motion_file_names = []
        for i in range(self._num_obj):
            obj_info_module[i] = os.path.join(project_dir,obj_info_module[i])
            spec = importlib.util.spec_from_file_location("obj_info%d"%(i),obj_info_module[i])
            obj_info = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(obj_info)
            self._obj_info.append(obj_info)
            obj_sim_agent = sim_agent.SimAgent(
                name="obj_sim_agent_%d"%(i),
                pybullet_client=self._pb_client, 
                model_file=sim_obj_file[i], 
                char_info=obj_info, 
                ref_scale=ref_motion_scale[i],
                ref_height_fix=0,
                self_collision=True,
                actuation="none",
                kinematic_only=False,
                verbose=self._verbose)

            obj_kin_agent = sim_agent.SimAgent(pybullet_client=self._base_env._pb_client, 
                            model_file=sim_obj_file[i],
                            char_info=obj_sim_agent._char_info,
                            ref_scale=obj_sim_agent._ref_scale,
                            ref_height_fix=obj_sim_agent._ref_height_fix,
                            self_collision=obj_sim_agent._self_collision,
                            reshape_config = obj_sim_agent._reshape_config,
                            kinematic_only=True,
                            verbose=self._base_env._verbose)

            self._obj_sim_agent.append(obj_sim_agent)
            self._obj_kin_agent.append(obj_kin_agent)

            ref_motion_all, ref_motion_file_names,ref_motion_file_asf_names= \
                motion_utils.load_motions(
                    ref_motion_file[i], 
                    None,
                    obj_sim_agent._char_info,
                    self._verbose)
                    
            self._obj_ref_motion_all.append(ref_motion_all)
            self._obj_ref_motion_file_names.append(ref_motion_file_names)

    def create(self):
        # if self._include_object:
        #     self.load_object()
            
        project_dir = self._config['project_dir']
        ref_motion_db = self._config['character'].get('ref_motion_db')
        ref_motion_file = motion_utils.collect_motion_files(project_dir, ref_motion_db)
        
        ''' Load Reference Motion '''

        self._ref_motion_all = []
        self._ref_motion_file_names = []
        self._sim_interaction_points = []
        self._kin_interaction_points = []
        self._constraints = [None]*len(self._target_constraints)
        for i in range(self._num_agent):
            ref_motion_all, ref_motion_file_names,ref_motion_file_asf_names= \
                motion_utils.load_motions(
                    ref_motion_file[i], 
                    None,
                    self._sim_agent[i]._char_info,
                    self._verbose)
            self._ref_motion_all.append(ref_motion_all)
            self._ref_motion_file_names.append(ref_motion_file_names)
            self._sim_interaction_points.append( None)
            self._kin_interaction_points.append( None)
            self._full_matrix_dist.append(None)
            self._full_matrix_dist_raw.append(None)
        """Load Saved Refernce Interaction Connectivity"""
        if self._load_ref_interaction is not None:
            ref_interaction_path = os.path.join(project_dir, self._load_ref_interaction)
            self._load_ref_interaction = pickle.load(open(ref_interaction_path,'rb'))
           
        ''' Load Probability of Motion Trajectory '''
        prob_trajectory = self._config['reward'].get('prob_trajectory')
        if prob_trajectory:
            # TODO: Load Model
            self._prob_trajectory = None

        '''Generate a template IG for the T-Pose'''
        # self.reset()
        
        self.reset_default_pose({})

        self._base_sim_ig = []
        self._base_kin_ig = []
        for idx in range(self._num_agent):
            sim_ig = self.compute_interaction_mesh(self._sim_interaction_points[idx])
            self._base_sim_ig.append(sim_ig)

            kin_ig = self.compute_interaction_mesh(self._kin_interaction_points[idx])
            self._base_kin_ig.append(kin_ig)

        ''' Should call reset after all setups are done '''

        self.reset({'add_noise': False})

        self._initialized = True

        if self._verbose:
            print('----- Humanoid Imitation Environment Created -----')
            for i in range(self._num_agent):
                print('[Agent%d]: state(%d) and action(%d)' \
                      %(i, self.dim_state(i), self.dim_action(i)))
            print('-------------------------------')
    def set_init_state_dist(self,dist):
        self._init_dist_bin = dist

    def reject_sampled_time(self,time):
        reject = False
        for reject_window in self._init_state_dist_reject:
            if time >= reject_window[0] and time <= reject_window[1]:
                reject= True
                break
        return reject
    def sample_start_time(self,ref_motion):
        if self._init_state_dist_type == 'uniform':
            time = np.random.uniform(0.0, ref_motion[0].length())

        elif self._init_state_dist_type == 'bin':
            if self._init_dist_bin is None:
                l = int(ref_motion[0].length()*self._config['fps_con'])
                dist = np.ones(l)/l
            else:
                dist = np.array(self._init_dist_bin)

            sample_reject = True
            while sample_reject:
                frame = np.random.choice(dist.shape[0],p=dist)
                time = frame/self._config['fps_con']
                sample_reject = self.reject_sampled_time(time)

            
        for i in range(self._num_agent):
            self._start_time[i] = time 
          
        self.episode_start_time = time
        horizon_duration = self._config['horizon']/self._config['fps_con']
        self.episode_expected_duration = min(ref_motion[0].length()-time,horizon_duration) 
    def callback_reset_prev(self, info):

        ''' Choose a reference motion randomly whenever reset '''
        
        self._ref_motion, self._ref_motion_idx = \
            self.sample_ref_motion(info.get('ref_motion_id'))
        
        ''' Choose a start time for the current reference motion '''
        
        if 'start_time' in info.keys():
            self._start_time = np.array(info.get('start_time'))
            assert self._start_time.shape[0] == self._num_agent
        else:
            self.sample_start_time(self._ref_motion)
        
        if self._include_object:
            self._obj_ref_motion = []
            for i in range(self._num_obj):
                self._obj_ref_motion.append(self._obj_ref_motion_all[i][self._ref_motion_idx[i]])
            self._obj_init_poses, self._obj_init_vels = self.set_object_pose_vel(info)

    def remove_neighbour_connectivity(self):
        if not self._compute_interaction_connectivity:
            return
        s = self.get_all_interaction_points("sim",0).shape[0]
        edges = np.ones((s,s))
        if self._is_prune_edges:
            ## Right now doing manual picking on neighbour edges
            edges[1,2] , edges[2,1] = 0,0
            edges[3,4] , edges[4,3] = 0,0
            edges[5,6] , edges[6,5] = 0,0
            edges[7,8] , edges[8,7] = 0,0


        return edges
        
    def callback_reset_after(self, info):
        self.current_interaction = {}
        for i,_ in enumerate(self._constraints):
            if self._constraints[i] is not None:
                self._pb_client.removeConstraint(self._constraints[i])
                self._constraints[i] = None

        for i in range(self._num_agent):
            self._kin_agent[i].set_pose(
                self._init_poses[i], self._init_vels[i])
        if self._include_object:
            for i in range(self._num_obj):
                self._obj_kin_agent[i].set_pose(
                    self._obj_init_poses[i], self._obj_init_vels[i])       
        for i in range(self._num_agent):
            """Recompute cartesian states of all interaction points"""
            self._sim_interaction_points[i] = self.get_all_interaction_points("sim",i)
            self._kin_interaction_points[i] = self.get_all_interaction_points("kin",i)

            """Compute/Load interaction connectivity"""
            if self._compute_interaction_connectivity:
                interaction_type = None
                if info.get("interaction_type"):
                    interaction_type = info.get('interaction_type')
                edge_indices = self.compute_reference_interaction_mesh("kin",i,interaction_type=interaction_type)

                self.current_interaction[i] = edge_indices
    def callback_step_prev(self, actions, infos):
        if self._num_agent==2 and self._target_constraints:
            constraint_eps = 1e-6
            curr_time = self.get_current_time()[0]
            for i, target_constraint in enumerate(self._target_constraints):
                start_time = target_constraint['start_time']
                duration = target_constraint['duration']
                parent_agent_id =target_constraint['parent_agent']
                parent_sim_agent = self._sim_agent[parent_agent_id]
                parent_sim_link = parent_sim_agent._char_info.joint_idx[target_constraint['parent_link']]
                parent_kin_agent = self._kin_agent[parent_agent_id]
                parent_kin_link = parent_kin_agent._char_info.joint_idx[target_constraint['parent_link']]

                child_sim_agent = self._sim_agent[target_constraint['child_agent']]
                child_sim_link = child_sim_agent._char_info.joint_idx[target_constraint['child_link']]

                child_kin_agent = self._kin_agent[target_constraint['child_agent']]
                child_kin_link = child_kin_agent._char_info.joint_idx[target_constraint['child_link']]

                parent_sim_link_pos,parent_sim_link_Q,_,_ = parent_sim_agent.get_link_states([parent_sim_link])
                child_sim_link_pos,child_sim_link_Q,_,_ = child_sim_agent.get_link_states([child_sim_link])

            
                parent_kin_link_pos,_,_,_ = parent_kin_agent.get_link_states([parent_kin_link])
                child_kin_link_pos,_,_,_ = child_kin_agent.get_link_states([child_kin_link])

                sim_dist = np.linalg.norm(parent_sim_link_pos - child_sim_link_pos)
                kin_dist = np.linalg.norm(parent_kin_link_pos - child_kin_link_pos)
                if (sim_dist <= kin_dist+constraint_eps) and curr_time >= start_time and curr_time< start_time+duration and (self._constraints[i] is None):
                    child_R = conversions.Q2R(child_sim_link_Q[0])
                    parent_R = conversions.Q2R(parent_sim_link_Q[0])
                    parent_R_inv = parent_R.transpose()
                    child_pos_in_parent_frame = parent_R_inv@(child_sim_link_pos[0]-parent_sim_link_pos[0])
                    constraint = self._pb_client.createConstraint(
                        parent_sim_agent._body_id,
                        parent_sim_link,
                        child_sim_agent._body_id,
                        child_sim_link,
                        self._pb_client.JOINT_FIXED,
                        np.zeros(3),
                        parentFramePosition=child_pos_in_parent_frame,
                        childFramePosition = np.zeros(3),
                        parentFrameOrientation = conversions.R2Q(parent_R_inv@child_R),
                        childFrameOrientation= conversions.R2Q(constants.EYE_R)
                        )
                    self._constraints[i] = constraint

                if curr_time >= start_time + duration and (self._constraints[i] is not None):
                    self._pb_client.removeConstraint(self._constraints[i])
                    self._constraints[i] = None
    def callback_step_after(self, actions, infos):
        ''' This is necessary to compute the reward correctly '''
        time = self.get_ref_motion_time()
        self.current_interaction = {}
        for i in range(self._num_agent):
            self._kin_agent[i].set_pose(
                self._ref_motion[i].get_pose_by_time(time[i]),
                self._ref_motion[i].get_velocity_by_time(time[i]))
        if self._include_object:
            for i in range(self._num_obj):
                self._obj_kin_agent[i].set_pose(
                    self._obj_ref_motion[i].get_pose_by_time(time[i]),
                    self._obj_ref_motion[i].get_velocity_by_time(time[i]))      
                  
        for i in range(self._num_agent):
            """Recompute cartesian states of all interaction points"""
            self._sim_interaction_points[i] = self.get_all_interaction_points("sim",i)
            self._kin_interaction_points[i] = self.get_all_interaction_points("kin",i)

            """Compute/Load interaction connectivity"""
            if self._compute_interaction_connectivity:
                edge_indices = self.compute_reference_interaction_mesh("kin",i)

                self.current_interaction[i] = edge_indices
        for i in range(self._num_agent):
            infos[i]['episode_start_time'] = self.episode_start_time
            infos[i]['episode_expected_duration'] = self.episode_expected_duration
            infos[i]['episode_current_time']=self.get_current_time()[i]
            infos[i]['episode_full_length']=self._ref_motion[0].length()
    def print_log_in_step(self):
        if self._verbose and self._end_of_episode:
            print('=================EOE=================')
            print('Reason:', self._end_of_episode_reason)
            print('TIME: (start:%02f) (elapsed:%02f) (time_after_eoe: %02f)'\
                %(self._start_time,
                  self.get_elapsed_time(),
                  self._time_elapsed_after_end_of_episode))
            print('=====================================')
    def compute_reference_interaction_mesh(self,agent_type, idx,interaction_type=None):

        if self._load_ref_interaction:
            motion_idx = self._ref_motion_idx[idx]
            current_time = self.get_ref_motion_time()
            current_frame = self._ref_motion[idx].time_to_frame(current_time)
            current_frame = min(current_frame,self._ref_motion[idx].num_frames()-1)
            edge_indices = self._load_ref_interaction[motion_idx][current_frame][idx]
            return edge_indices
        else:
            if interaction_type is None:
                interaction_type = self._interaction_type

            all_points = []
            interaction_points = self.get_all_interaction_points(agent_type,idx)

            all_points.append(interaction_points[:,:3])
            all_points = np.concatenate(all_points,axis=0)
            self_cnt = 0
            if "self" in self._interaction_choices:
                self_cnt = len(self._interaction_joints)
            interaction = Interaction(all_points,self_cnt)

            edge_indices = interaction.build_interaction_graph(interaction_type)

            return edge_indices

    def get_all_interaction_points_name(self,*args):

        interaction_joints = self._config['interaction'].get("interaction_joint_candidates",[])
        oppo_interaction_joints = self._config['interaction'].get("oppo_interaction_joint_candidates",[])

        self_interaction_name = []
        oppo_interaction_name = []
        for i in range(len(interaction_joints)):
            name = interaction_joints[i] 
            if type(name) is dict:
                name = list(name.keys())[0]
            self_interaction_name.append(name)

        for i in range(len(oppo_interaction_joints)):
            name = oppo_interaction_joints[i] 
            if type(name) is dict:
                name = list(name.keys())[0]
            oppo_interaction_name.append(name)          


        interaction_points_name =  list(self_interaction_name + oppo_interaction_name)
        return interaction_points_name
    def get_all_interaction_points(self,agent_type,idx,*args):
        if agent_type == "sim":
            agent = self._sim_agent[idx]
            if len(self._sim_agent)>1:
                other_agent = self._sim_agent[1-idx]
            if self._include_object:
                obj_agent = self._obj_sim_agent[0]
        elif agent_type == "kin":
            agent = self._kin_agent[idx]
            if len(self._kin_agent)>1:
                other_agent = self._kin_agent[1-idx]
            if self._include_object:
                obj_agent = self._obj_kin_agent[0]
        interaction_points = []

        if "self" in self._interaction_choices:
            if self._interaction_joints is not None:
                joint_candidate = np.array(self._interaction_joints)
            else:
                joint_candidate = np.array(agent._char_info.interaction_joint_candidate)
            joint_state = np.array(agent.get_joint_cartesian_state_fast(np.array(joint_candidate),offsets=np.array(self._interaction_joints_xform),index_type=self._index_type))
            interaction_points.append(joint_state)

        if "ground" in self._interaction_choices:
            T = agent.get_facing_transform(self.get_ground_height(idx))
            _, _, v, _ = agent.get_root_state()
            R,p = conversions.T2Rp(T)
            v_facing = v - math.projectionOnVector(v, agent._char_info.v_up_env)

            ground_point = np.zeros((1,6))
            ground_point[0,:3] = p
            ground_point[0,3:6] = v_facing

            interaction_points.append(ground_point)

        if "object" in self._interaction_choices:
            vert_candidate = np.array(self._object_interaction_joints)

            joint_state = np.array(obj_agent.get_joint_cartesian_state_fast(np.array(vert_candidate),offsets=np.array(self._object_interaction_joints_xform),index_type=self._index_type))
            interaction_points.append(joint_state)   

        ## This block MUST stay at the end to make sure the reward function works...
        if "other_agent" in self._interaction_choices:
            if self._oppo_interaction_joints is not None:
                joint_candidate = np.array(self._oppo_interaction_joints)
            else:
                joint_candidate = np.array(other_agent._char_info.interaction_joint_candidate)
            joint_state = np.array(other_agent.get_joint_cartesian_state_fast(np.array(joint_candidate),offsets=np.array(self._oppo_interaction_joints_xform),index_type=self._index_type))
            interaction_points.append(joint_state)

        return np.concatenate(interaction_points,axis=0)

    def compute_distance_weights(self,idx,edge_index=None,return_dist = False,weight_type='kin'):
        if weight_type == 'kin':
            points = self._kin_interaction_points[idx]
        
            mesh = self.compute_interaction_mesh(points,edge_index)
            if(len(mesh.shape))==3:
                """If the mesh is a dense matrix"""
                dist = np.linalg.norm(mesh[:,:,:3],axis=2)
            else:
                """If the mesh is a sparse matrix"""
                dist = np.linalg.norm(mesh[:,:3],axis=1)
            
            weights = np.exp(-self._interaction_kernel*dist)
            weights = weights/np.sum(weights)
            if return_dist:
                return weights,dist
            else:
                return weights 
        elif weight_type == 'kin+sim':
            kin_points = self._kin_interaction_points[idx]
            sim_points = self._sim_interaction_points[idx]
        
            kin_mesh = self.compute_interaction_mesh(kin_points,edge_index)
            sim_mesh = self.compute_interaction_mesh(sim_points,edge_index)
            if(len(kin_mesh.shape))==3:
                """If the mesh is a dense matrix"""
                kin_dist = np.linalg.norm(kin_mesh[:,:,:3],axis=2)
                sim_dist = np.linalg.norm(sim_mesh[:,:,:3],axis=2)

            else:
                """If the mesh is a sparse matrix"""
                kin_dist = np.linalg.norm(kin_mesh[:,:3],axis=1)
                sim_dist = np.linalg.norm(sim_mesh[:,:3],axis=1)

            
            kin_weights = np.exp(-self._interaction_kernel*kin_dist)
            sim_weights = np.exp(-self._interaction_kernel*sim_dist)

            weights = 0.5*kin_weights/np.sum(kin_weights) + 0.5*sim_weights/np.sum(sim_weights)
            if return_dist:
                return weights,0.5*(sim_dist+kin_dist)
            else:
                return weights  
        else:
            raise NotImplementedError   
    # def compute_distance_weights(self,points,edge_index=None,return_dist = False):
    #     mesh = self.compute_interaction_mesh(points,edge_index)
    #     if(len(mesh.shape))==3:
    #         """If the mesh is a dense matrix"""
    #         dist = np.linalg.norm(mesh[:,:,:3],axis=2)
    #     else:
    #         """If the mesh is a sparse matrix"""
    #         dist = np.linalg.norm(mesh[:,:3],axis=1)
        
    #     weights = np.exp(-self._interaction_kernel*dist)
    #     weights = weights/np.sum(weights)
    #     if return_dist:
    #         return weights,dist
    #     else:
    #         return weights
    def compute_interaction_mesh(self,points,edge_index=None,T=constants.EYE_T):

        R,p = conversions.T2Rp(T)
        R_inv = R.transpose()
        # interaction_candidate1 = agent1._char_info.interaction_joint_candidate
        # interaction_candidate2 = agent2._char_info.interaction_joint_candidate

        # joint_state1 = agent1.get_joint_cartesian_state()
        # joint_state2 = agent2.get_joint_cartesian_state()


        # '''Interaction mesh Position'''
        # for i in interaction_candidate1:
        #     for j in interaction_candidate2:
        #         pi,vi = joint_state1[i+1][:3],joint_state1[i+1][3:]
        #         pj,vj = joint_state2[j+1][:3],joint_state2[j+1][3:]

        #         dp = R @ ((pj - pi) - p)
        #         dv = R @ (vj - vi)
        #         mesh_data[i+1,j+1] = np.hstack([dp,dv])

        # end_bf = time.time()
        # print ("Brute Force Time elapsed:", end_bf - start_bf)

        # joint_state1_np = np.array(joint_state1)
        # joint_state2_np = np.array(joint_state2)
        # interaction_candidate1_np = np.array(interaction_candidate1)
        # interaction_candidate2_np = np.array(interaction_candidate2)

        # int_i = joint_state1_np[interaction_candidate1_np+1]
        # int_j = joint_state2_np[interaction_candidate2_np+1]

        int_i = np.array(points)
        int_j = np.array(points)
        if edge_index is None:
            int_i_exp = np.expand_dims(int_i,1)
            int_i_repeat = np.repeat(int_i_exp,len(points),1)
            int_j_exp = np.expand_dims(int_j,0)
            int_j_repeat = np.repeat(int_j_exp,len(points),0)

            d_int = (int_j_repeat - int_i_repeat)

            d_int  [:,:,:3]=  (R_inv @ d_int[:,:,:3].transpose(0,2,1)).transpose(0,2,1)
            d_int  [:,:,3:]= (R_inv @ d_int[:,:,3:].transpose(0,2,1)).transpose(0,2,1)
            return d_int
        else:
            int_i = int_i[edge_index[0]]
            int_j = int_j[edge_index[1]]

            d_int = int_j - int_i
            
            d_int[:,:3] = (R_inv @ d_int[:,:3].T).T
            d_int [:,3:]= (R_inv @ d_int[:,3:].T).T
            return d_int

    def set_object_pose_vel(self,info):
        init_poses, init_vels = [], []
        for i in range(len(self._obj_sim_agent)):
            init_pose = self._obj_ref_motion[i].get_pose_by_time(self._start_time[i])
            init_vel = self._obj_ref_motion[i].get_velocity_by_time(self._start_time[i])

            if info.get('add_noise'):
                init_pose, init_vel = \
                    self._base_env.add_noise_to_pose_vel(
                        self._obj_sim_agent[i], init_pose, init_vel)

        
            self._obj_sim_agent[i].set_pose(init_pose,init_vel)
            init_poses.append(init_pose)
            init_vels.append(init_vel)
        return  init_poses, init_vels
    def get_kin_lowest_height(self,idx):
        kin_agent = self._kin_agent[idx]
        up = kin_agent._char_info.v_up_env
        link_states_p,_,_,_ = kin_agent.get_link_states()
        heights = np.sum(link_states_p * up,axis=1)
        
        link_id, link_height = np.argmin(heights),np.min(heights)
        return link_id,link_height
    def reset_default_pose(self,info):

        self._init_poses, self._init_vels = self.compute_base_init_pose_vel(info)
        
        self.callback_reset_prev(info)

        self._base_env.reset(time=0.0,
                             poses=self._init_poses, 
                             vels=self._init_vels)

        self.callback_reset_after(info)

        self._rew_data_prev = \
            [self.reward_data(i) for i in range(self._num_agent)]

    def compute_base_init_pose_vel(self,info):
        ''' This performs reference-state-initialization (RSI) '''
        init_poses, init_vels = [], []

        for i in range(self._num_agent):
            ''' Set the state of simulated agent by using the state of reference motion '''

            init_pose = self._base_motion[i].get_pose_by_time(0)
            init_vel = self._base_motion[i].get_velocity_by_time(0)
            ''' Add noise to the state if necessary '''
            if info.get('add_noise'):
                init_pose, init_vel = \
                    self._base_env.add_noise_to_pose_vel(
                        self._sim_agent[i], init_pose, init_vel)
            init_poses.append(init_pose)
            init_vels.append(init_vel)
            
            self._kin_agent[i].set_pose(init_pose, None)
            link_id, link_height = self.get_kin_lowest_height(i)
            self._sim_agent[i]._ref_height_fix = self._kin_agent[i]._ref_height_fix
            self._sim_agent[i]._ref_height_fix_v = self._sim_agent[i]._ref_height_fix * self._sim_agent[i]._char_info.v_up_env            
            
            self._sim_agent[i].set_pose(init_pose, None)

            p,_,_,_ = self._sim_agent[i].get_link_states(indices=[link_id])
            hight_fix_v = (link_height-p)* self._sim_agent[i]._char_info.v_up_env

            self._sim_agent[i]._ref_height_fix = np.sum(hight_fix_v)
            self._sim_agent[i]._ref_height_fix_v = self._sim_agent[i]._ref_height_fix * self._sim_agent[i]._char_info.v_up_env
        
        
        return init_poses, init_vels

    def compute_init_pose_vel(self, info):
        ''' This performs reference-state-initialization (RSI) '''
        init_poses, init_vels = [], []

        for i in range(self._num_agent):
            ''' Set the state of simulated agent by using the state of reference motion '''

            init_pose = self._ref_motion[i].get_pose_by_time(self._start_time[i])
            init_vel = self._ref_motion[i].get_velocity_by_time(self._start_time[i])
            
            ''' Add noise to the state if necessary '''
            if info.get('add_noise'):
                init_pose, init_vel = \
                    self._base_env.add_noise_to_pose_vel(
                        self._sim_agent[i], init_pose, init_vel)
            init_poses.append(init_pose)
            init_vels.append(init_vel)
            
            self._kin_agent[i].set_pose(init_pose, None)
            link_id, link_height = self.get_kin_lowest_height(i)
            self._sim_agent[i]._ref_height_fix = self._kin_agent[i]._ref_height_fix
            self._sim_agent[i]._ref_height_fix_v = self._sim_agent[i]._ref_height_fix * self._sim_agent[i]._char_info.v_up_env            
            
            self._sim_agent[i].set_pose(init_pose, None)

            p,_,_,_ = self._sim_agent[i].get_link_states(indices=[link_id])
            hight_fix_v = (link_height-p)* self._sim_agent[i]._char_info.v_up_env

            self._sim_agent[i]._ref_height_fix = np.sum(hight_fix_v)
            self._sim_agent[i]._ref_height_fix_v = self._sim_agent[i]._ref_height_fix * self._sim_agent[i]._char_info.v_up_env
        
        
        return init_poses, init_vels
    def get_general_interaction_graph(self,idx,agent_type="sim"):
        if agent_type == "sim":
            agent = self._sim_agent[idx]
            interaction_point_states = np.array(self._sim_interaction_points[idx])
        elif agent_type == "kin":
            agent = self._kin_agent[idx]
            interaction_point_states = np.array(self._kin_interaction_points[idx])
        else:
            raise NotImplementedError

        T = agent.get_facing_transform(self.get_ground_height(idx))
        R,p = conversions.T2Rp(T)
        R_inv = R.transpose()
        interaction_point_states[:,:3] = interaction_point_states[:,:3] - p
        interaction_point_states[:,:3] =  (R_inv@interaction_point_states[:,:3].transpose()).transpose()
        interaction_point_states[:,3:]=  (R_inv@interaction_point_states[:,3:].transpose()).transpose()

        edge_indices = self.current_interaction[idx]
        
        if self._sparse_interaction:
            edges = self.compute_interaction_mesh(interaction_point_states,edge_index=edge_indices)
            edge_index_full_holder = np.zeros((2,interaction_point_states.shape[0]*interaction_point_states.shape[0]))
            edge_attr_full_holder = np.zeros((interaction_point_states.shape[0]*interaction_point_states.shape[0],edges.shape[1]))
            edge_index_full_holder[:,:edge_indices[0].shape[0]] = edge_indices[:,:]
            edge_attr_full_holder[:edge_indices[0].shape[0],:] = edges
            graph_data = [interaction_point_states.ravel(),edge_indices[0].shape[0],edge_index_full_holder.ravel(),edge_attr_full_holder.ravel()]            
        else:
            row = edge_indices[0]
            col = edge_indices[1]
            data = np.ones_like(row)
            edge_indices_mtx = coo_matrix((data,(row,col)),shape=(len(interaction_point_states),len(interaction_point_states))).todense()
            edge_indices_mtx = np.expand_dims(edge_indices_mtx,axis=2)
            edge_attr = self.compute_interaction_mesh(interaction_point_states)
            edges = edge_indices_mtx * edge_attr
            
            graph_data = [interaction_point_states.ravel(),edges.ravel()]
        return np.hstack(graph_data)
    def get_state_by_key(self, idx, key):
        state = []

        ref_motion = self._ref_motion[idx] 
        time = self.get_ref_motion_time()
        if key=='body':
            state.append(self.state_body(idx, "sim"))
        elif key == 'oppo_body':
            oppo_idx = 1-idx
            state.append(self.state_body(oppo_idx, "sim",type="oppo_facing_R6_h"))
        elif key=='ref_motion_abs' or key=='ref_motion_rel' or key=='ref_motion_abs_rel':
            ref_motion_abs = True \
                if (key=='ref_motion_abs' or key=='ref_motion_abs_rel') else False
            ref_motion_rel = True \
                if (key=='ref_motion_rel' or key=='ref_motion_abs_rel') else False
            poses, vels = [], []
            for dt in self._sensor_lookahead:
                t = np.clip(
                    time[idx] + dt, 
                    0.0, 
                    ref_motion.length())
                poses.append(ref_motion.get_pose_by_time(t))
                vels.append(ref_motion.get_velocity_by_time(t))
            state.append(self.state_imitation(idx,
                                              poses,
                                              vels,
                                              include_abs=ref_motion_abs,
                                              include_rel=ref_motion_rel))
        elif key=="oppo_ref_motion_abs":
            oppo_idx = 1-idx
            oppo_ref_motion = self._ref_motion[oppo_idx]
            poses, vels = [], []
            for dt in self._sensor_lookahead:
                t = np.clip(
                    time[oppo_idx] + dt, 
                    0.0, 
                    oppo_ref_motion.length())
                poses.append(oppo_ref_motion.get_pose_by_time(t))
                vels.append(oppo_ref_motion.get_velocity_by_time(t))
            state.append(self.state_imitation(oppo_idx,
                                              poses,
                                              vels,
                                              include_abs=True,
                                              include_rel=False,
                                              type="oppo_facing_R6_h"))

        elif key=="sim_interaction_graph_state":
            start_time = tt.time()

            interaction_graph = self.get_general_interaction_graph(idx,agent_type="sim")
            state.append(np.hstack(interaction_graph))
            # print("---Simulation State Execution: %.4f seconds ---" % (tt.time() - start_time))

        elif key=="ref_interaction_graph_state":
            start_time = tt.time()

            poses, vels = [], []
            for dt in self._sensor_lookahead:
                t = np.clip(
                    time[idx] + dt, 
                    0.0, 
                    ref_motion.length())
                poses.append(ref_motion.get_pose_by_time(t))
                vels.append(ref_motion.get_velocity_by_time(t))
            kin_joint_states = self.state_imitation_interaction_graph_state(
                idx,
                poses,
                vels,
                only_height=True
            )
            state.append(kin_joint_states.flatten())
            # print("---Reference State Execution: %.4f seconds ---" % (tt.time() - start_time))

        elif key == 'sim_interaction_points_height':
            sim_agent = self._sim_agent[idx]
            inter_point_height = self._sim_interaction_points[idx][:,:3] * sim_agent._char_info.v_up_env
            inter_point_height = np.sum(inter_point_height,axis=1)
            state.append(np.hstack(inter_point_height))
        elif key == "ref_interaction_points_height":
            sim_agent = self._sim_agent[idx]
            poses, vels = [], []
            for dt in self._sensor_lookahead:
                t = np.clip(
                    time[idx] + dt, 
                    0.0, 
                    ref_motion.length())
                poses.append(ref_motion.get_pose_by_time(t))
                vels.append(ref_motion.get_velocity_by_time(t))
            kin_joint_states = self.state_imitation_interaction_graph_state(
                idx,
                poses,
                vels,
                only_height=True,
                only_points= True
            )
            state.append(kin_joint_states)    
        elif key == 'obj_body':
            for obj_idx in range(self._num_obj):
                state.append(self.state_object(idx,obj_idx, "sim"))
        elif key == 'ref_obj_abs':
            char_poses, char_vels, obj_poses,obj_vels = [], [], [], []

            for obj_idx in range(self._num_obj):
                obj_ref_motion = self._obj_ref_motion[obj_idx]
                for dt in self._sensor_lookahead:
                    t = np.clip(
                        time[idx] + dt, 
                        0.0, 
                        ref_motion.length())
                    
                    char_poses.append(ref_motion.get_pose_by_time(t))
                    char_vels.append(ref_motion.get_velocity_by_time(t))

                    obj_poses.append(obj_ref_motion.get_pose_by_time(t))
                    obj_vels.append(obj_ref_motion.get_velocity_by_time(t))

                state.append(self.obj_state_imitation(idx,
                                                obj_idx,
                                                obj_poses,
                                                obj_vels,
                                                char_poses,
                                                char_vels))
        else:
            raise NotImplementedError

        return np.hstack(state)
    def dim_feature(self,idx):
        return (self.get_all_interaction_points("sim",idx).shape[0],6)
    def dim_interaction(self,idx):
        if self.current_interaction:
            return self.get_general_interaction_graph(0,"sim").shape[idx]
        else:
            return 0

    
    def num_interaction(self,idx):
        sc = self._state_choices.copy()
        num = 0
        for i in sc:
            if "sim_interaction_graph_state" == i :
                num+=1
            elif "ref_interaction_graph_state" == i:
                num += len(self._sensor_lookahead)
        return num
    def is_sparse_interaction(self):
        return self._sparse_interaction
    def state_imitation_interaction_graph_state(self,idx,poses,vels,only_height = False, only_points = False):
        assert len(poses) == len(vels)
        kin_agent = self._kin_agent[idx]

        state = []
        state_kin_orig = kin_agent.save_states()
        for pose, vel in zip(poses, vels):
            kin_agent.set_pose(pose, vel)
            self._kin_interaction_points[idx] = self.get_all_interaction_points("kin",idx)
            if only_points:
                inter_points = self.get_all_interaction_points("kin",idx)
                if only_height:
                    inter_points = inter_points[:,:3] * kin_agent._char_info.v_up_env
                    inter_points = np.sum(inter_points,axis=1)
                state.append(np.hstack(inter_points))
            else:
                inter_graph = self.get_general_interaction_graph(idx,agent_type="kin")
                state.append(np.hstack(inter_graph))
            
        kin_agent.restore_states(state_kin_orig)
        self._kin_interaction_points[idx] = self.get_all_interaction_points("kin",idx)

        return np.hstack(state)
    def state_object(
        self, 
        idx, 
        obj_idx,
        agent='sim',
        type=None, 
        return_stacked=True):
        if agent == "sim":
            obj_agent = self._obj_sim_agent[obj_idx]
            char_agent = self._sim_agent[idx]
        elif agent == "kin":
            obj_agent = self._obj_kin_agent[obj_idx]
            char_agent = self._kin_agent[idx]
        else:
            raise NotImplementedError

        return self._state_object(idx,obj_agent,char_agent, type, return_stacked)

    def state_body(
        self, 
        idx, 
        agent="sim", 
        type=None, 
        return_stacked=True):
        oppo_agent = None
        if agent == "sim":
            agent = self._sim_agent[idx]
            if self._num_agent==2:
                oppo_agent = self._sim_agent[1-idx]
        elif agent == "kin":
            agent = self._kin_agent[idx]
            if self._num_agent==2:
                oppo_agent = self._kin_agent[1-idx]
        else:
            raise NotImplementedError

        return self._state_body(idx, agent, type, return_stacked,oppo_agent)

    def state_task(self, idx):
        sc = self._state_choices.copy()
        if 'body' in sc:
            sc.remove('body')
        return self.state(idx, sc)

    def obj_state_imitation(
        self, 
        idx,
        obj_idx,
        obj_poses, 
        obj_vels, 
        char_poses,
        char_vels,
        type=None):

        assert len(obj_poses) == len(obj_vels)

        obj_kin_agent = self._obj_kin_agent[obj_idx]
        char_kin_agent = self._kin_agent[idx]

        state = []
        obj_state_kin_orig = obj_kin_agent.save_states()
        char_state_kin_orig = char_kin_agent.save_states()

        for obj_pose, obj_vel,char_pose,char_vel in zip(obj_poses, obj_vels,char_poses,char_vels):
            obj_kin_agent.set_pose(obj_pose, obj_vel)
            char_kin_agent.set_pose(char_pose,char_vel)

            state_kin = self.state_object(idx,obj_idx, agent="kin", type=type,return_stacked=False)
            # Add pos/vel values
            state.append(np.hstack(state_kin))

            # ''' Add facing frame differences '''
            # R_kin, p_kin = conversions.T2Rp(
            #     kin_agent.get_facing_transform(self.get_ground_height(idx)))
            # state.append(np.dot(R_sim_inv, p_kin - p_sim))
            # state.append(np.dot(R_sim_inv, kin_agent.get_facing_direction()))
        obj_kin_agent.restore_states(obj_state_kin_orig)
        char_kin_agent.restore_states(char_state_kin_orig)

        
        return np.hstack(state)
    def state_imitation(
        self, 
        idx,
        poses, 
        vels, 
        include_abs, 
        include_rel,
        type=None):

        assert len(poses) == len(vels)

        sim_agent = self._sim_agent[idx]
        kin_agent = self._kin_agent[idx]

        R_sim, p_sim = conversions.T2Rp(
            sim_agent.get_facing_transform(self.get_ground_height(idx)))
        R_sim_inv = R_sim.transpose()
        state_sim = self.state_body(idx, agent="sim", type=type,return_stacked=False)
        
        state = []
        state_kin_orig = kin_agent.save_states()
        for pose, vel in zip(poses, vels):
            kin_agent.set_pose(pose, vel)
            state_kin = self.state_body(idx, agent="kin", type=type,return_stacked=False)
            # Add pos/vel values
            if include_abs:
                state.append(np.hstack(state_kin))
            # Add difference of pos/vel values
            if include_rel:
                for j in range(len(state_sim)):
                    if np.isscalar(state_sim[j]) or len(state_sim[j])==3 or len(state_sim[j])==6: 
                        state.append(state_sim[j]-state_kin[j])
                    elif len(state_sim[j])==4:
                        state.append(
                            self._pb_client.getDifferenceQuaternion(state_sim[j], state_kin[j]))
                    else:
                        raise NotImplementedError
            ''' Add facing frame differences '''
            R_kin, p_kin = conversions.T2Rp(
                kin_agent.get_facing_transform(self.get_ground_height(idx)))
            state.append(np.dot(R_sim_inv, p_kin - p_sim))
            state.append(np.dot(R_sim_inv, kin_agent.get_facing_direction()))
        kin_agent.restore_states(state_kin_orig)

        return np.hstack(state)

    def reward_data(self, idx):
        data = {}

        data['sim_root_pQvw'] = self._sim_agent[idx].get_root_state()
        # data['sim_interaction_graph'] = self.compute_interaction_mesh_lower_triangle(self._sim_agent[idx])

        data['sim_link_pQvw'] = self._sim_agent[idx].get_link_states()
        data['sim_joint_pv'] = self._sim_agent[idx].get_joint_states()
        data['sim_facing_frame'] = self._sim_agent[idx].get_facing_transform(self.get_ground_height(idx))
        data['sim_com'], data['sim_com_vel'] = self._sim_agent[idx].get_com_and_com_vel()
        
        data['kin_root_pQvw'] = self._kin_agent[idx].get_root_state()
        # data['kin_interaction_graph'] = self.compute_interaction_mesh_lower_triangle(self._kin_agent[idx])

        data['kin_link_pQvw'] = self._kin_agent[idx].get_link_states()
        data['kin_joint_pv'] = self._kin_agent[idx].get_joint_states()
        data['kin_facing_frame'] = self._kin_agent[idx].get_facing_transform(self.get_ground_height(idx))
        data['kin_com'], data['kin_com_vel'] = self._kin_agent[idx].get_com_and_com_vel()

        if self._compute_interaction_connectivity:
            edge_indices = self.current_interaction[idx]
            row = edge_indices[0]
            col = edge_indices[1]
            edge_data = np.ones_like(row)
            edge_indices_mtx = coo_matrix((edge_data,(row,col)),shape=(len(self._sim_interaction_points[0]),len(self._sim_interaction_points[0]))).toarray()
            edge_indices_mtx = coo_matrix(edge_indices_mtx * self._prune_edges)
            edge_indices = np.array([edge_indices_mtx.row,edge_indices_mtx.col])
        else:
            edge_indices = np.array([[0],[0]])
        data['edge_indices'] = edge_indices.copy()
        data['sim_sim_interaction_graph'] = self.compute_interaction_mesh(self._sim_interaction_points[idx],edge_indices)
        data['kin_kin_interaction_graph'] = self.compute_interaction_mesh(self._kin_interaction_points[idx],edge_indices)
        data['kin_interaction_dist_weight'] = self.compute_distance_weights(idx,edge_indices,weight_type=self._interaction_weight_type)
        
        if self._include_object:
            data['obj_sim_root_pQvw'] =self._obj_sim_agent[0].get_root_state()
            data['obj_kin_root_pQvw'] =self._obj_kin_agent[0].get_root_state()

            data['obj_sim_com'],data['obj_sim_com_vel']  = self._obj_sim_agent[0].get_com_and_com_vel()
            data['obj_kin_com'],data['obj_kin_com_vel'] = self._obj_kin_agent[0].get_com_and_com_vel()

        return data 
    
    def reward_max(self):
        return 1.0
    
    def reward_min(self):
        return 0.0
    
    def get_task_error(self, idx, data_prev, data_next, actions):
        error = {}

        sim_agent = self._sim_agent[idx]
        char_info = sim_agent._char_info

        data = data_next[idx]

        sim_root_p, sim_root_Q, sim_root_v, sim_root_w = data['sim_root_pQvw']
        sim_link_p, sim_link_Q, sim_link_v, sim_link_w = data['sim_link_pQvw']
        sim_joint_p, sim_joint_v = data['sim_joint_pv']
        sim_facing_frame = data['sim_facing_frame']
        # sim_iteration_graph = data['sim_interaction_graph']

        R_sim_f, p_sim_f = conversions.T2Rp(sim_facing_frame)
        R_sim_f_inv = R_sim_f.transpose()
        sim_com, sim_com_vel = data['sim_com'], data['sim_com_vel']
        
        kin_root_p, kin_root_Q, kin_root_v, kin_root_w = data['kin_root_pQvw']
        kin_link_p, kin_link_Q, kin_link_v, kin_link_w = data['kin_link_pQvw']
        kin_joint_p, kin_joint_v = data['kin_joint_pv']
        kin_facing_frame = data['kin_facing_frame']
        # kin_iteration_graph = data['kin_interaction_graph']

        
        sim_sim_interaction_graph = data['sim_sim_interaction_graph']
        
        kin_kin_interaction_graph = data['kin_kin_interaction_graph']

        kin_dist_weights = data['kin_interaction_dist_weight']
        pruned_edges = data['edge_indices']
        
        R_kin_f, p_kin_f = conversions.T2Rp(kin_facing_frame)
        R_kin_f_inv = R_kin_f.transpose()
        kin_com, kin_com_vel = data['kin_com'], data['kin_com_vel']

        indices = range(len(sim_joint_p))
        if self._include_object:
            obj_sim_root_p, obj_sim_root_Q, obj_sim_root_v, obj_sim_root_w = data['obj_sim_root_pQvw']
            obj_kin_root_p, obj_kin_root_Q, obj_kin_root_v, obj_kin_root_w = data['obj_kin_root_pQvw']
            obj_sim_com, obj_sim_com_vel = data['obj_sim_com'], data['obj_sim_com_vel']
            obj_kin_com, obj_kin_com_vel = data['obj_kin_com'], data['obj_kin_com_vel']


        if self.exist_rew_fn_subterm(idx, 'pose_pos_nohand'):
            error['pose_pos_nohand'] = 0.0
            for j in indices:
                joint_type = sim_agent.get_joint_type(j)
                if joint_type == self._pb_client.JOINT_FIXED:
                    continue
                elif joint_type == self._pb_client.JOINT_SPHERICAL:
                    if j!= sim_agent._char_info.joint_idx['LeftHand'] and j!= sim_agent._char_info.joint_idx['RightHand']:
                        dQ = self._pb_client.getDifferenceQuaternion(sim_joint_p[j], kin_joint_p[j])
                        _, diff_pose_pos = self._pb_client.getAxisAngleFromQuaternion(dQ)
                    else:
                        continue
                else:
                    diff_pose_pos = sim_joint_p[j] - kin_joint_p[j]
                error['pose_pos_nohand'] += char_info.joint_weight[j] * np.dot(diff_pose_pos, diff_pose_pos)
            if len(indices) > 0:
                error['pose_pos_nohand'] /= len(indices)

        if self.exist_rew_fn_subterm(idx, 'pose_vel_nohand'):
            error['pose_vel_nohand'] = 0.0
            for j in indices:
                joint_type = sim_agent.get_joint_type(j)
                if joint_type == self._pb_client.JOINT_FIXED:
                    continue
                else:
                    if j!= sim_agent._char_info.joint_idx['LeftHand'] and j!= sim_agent._char_info.joint_idx['RightHand']:
                        diff_pose_vel = sim_joint_v[j] - kin_joint_v[j]
                    # print(idx,sim_agent._char_info.joint_name[j],np.dot(diff_pose_vel, diff_pose_vel))
                    # else:
                        # print("Hand Joint")
                        # continue
                error['pose_vel_nohand'] += char_info.joint_weight[j] * np.dot(diff_pose_vel, diff_pose_vel)
            if len(indices) > 0:
                error['pose_vel_nohand'] /= len(indices)

        if self.exist_rew_fn_subterm(idx, 'pose_pos'):
            error['pose_pos'] = 0.0
            for j in indices:
                joint_type = sim_agent.get_joint_type(j)
                if joint_type == self._pb_client.JOINT_FIXED:
                    continue
                elif joint_type == self._pb_client.JOINT_SPHERICAL:
                    
                    dQ = self._pb_client.getDifferenceQuaternion(sim_joint_p[j], kin_joint_p[j])
                    _, diff_pose_pos = self._pb_client.getAxisAngleFromQuaternion(dQ)

                else:
                    diff_pose_pos = sim_joint_p[j] - kin_joint_p[j]
                error['pose_pos'] += char_info.joint_weight[j] * np.dot(diff_pose_pos, diff_pose_pos)
            if len(indices) > 0:
                error['pose_pos'] /= len(indices)

        if self.exist_rew_fn_subterm(idx, 'pose_vel'):
            error['pose_vel'] = 0.0
            for j in indices:
                joint_type = sim_agent.get_joint_type(j)
                if joint_type == self._pb_client.JOINT_FIXED:
                    continue
                else:
                    diff_pose_vel = sim_joint_v[j] - kin_joint_v[j]
                error['pose_vel'] += char_info.joint_weight[j] * np.dot(diff_pose_vel, diff_pose_vel)
            if len(indices) > 0:
                error['pose_vel'] /= len(indices)

        if self.exist_rew_fn_subterm(idx, 'ee'):
            error['ee'] = 0.0
            
            for j in char_info.end_effector_indices:
                sim_ee_local = np.dot(R_sim_f_inv, sim_link_p[j]-p_sim_f)
                kin_ee_local = np.dot(R_kin_f_inv, kin_link_p[j]-p_kin_f)
                diff_pos =  sim_ee_local - kin_ee_local
                error['ee'] += np.dot(diff_pos, diff_pos)

            if len(char_info.end_effector_indices) > 0:
                error['ee'] /= len(char_info.end_effector_indices)
        if self.exist_rew_fn_subterm(idx, 'root_remove_height'):
            horizontal_vec = np.ones(3)-self._sim_agent[idx]._char_info.v_up_env
            diff_root_p = (sim_root_p - kin_root_p)*horizontal_vec
            _, diff_root_Q = self._pb_client.getAxisAngleFromQuaternion(
                self._pb_client.getDifferenceQuaternion(sim_root_Q, kin_root_Q))
            diff_root_v = (sim_root_v - kin_root_v)*horizontal_vec
            diff_root_w = (sim_root_w - kin_root_w)
            error['root_remove_height'] = 1.0 * np.dot(diff_root_p, diff_root_p) + \
                            0.1 * np.dot(diff_root_Q, diff_root_Q) + \
                            0.01 * np.dot(diff_root_v, diff_root_v) + \
                            0.001 * np.dot(diff_root_w, diff_root_w)

        if self.exist_rew_fn_subterm(idx, 'com_remove_height'):
            horizontal_vec = np.ones(3)-self._sim_agent[idx]._char_info.v_up_env
            diff_com = np.dot(R_sim_f_inv, sim_com*horizontal_vec-p_sim_f*horizontal_vec) - np.dot(R_kin_f_inv, kin_com*horizontal_vec-p_kin_f*horizontal_vec)
            diff_com_vel = np.dot(R_sim_f_inv, sim_com_vel*horizontal_vec) - np.dot(R_kin_f_inv, kin_com_vel*horizontal_vec)
            error['com_remove_height'] = 1.0 * np.dot(diff_com, diff_com) + \
                           0.1 * np.dot(diff_com_vel, diff_com_vel)

        if self.exist_rew_fn_subterm(idx, 'obj_root'):
            diff_root_p = obj_sim_root_p - obj_kin_root_p
            _, diff_root_Q = self._pb_client.getAxisAngleFromQuaternion(
                self._pb_client.getDifferenceQuaternion(obj_sim_root_Q, obj_kin_root_Q))
            diff_root_v = obj_sim_root_v - obj_kin_root_v
            diff_root_w = obj_sim_root_w - obj_kin_root_w

            error['obj_root'] = 1.0 * np.dot(diff_root_p, diff_root_p) + \
                            0.1 * np.dot(diff_root_Q, diff_root_Q) + \
                            0.01 * np.dot(diff_root_v, diff_root_v) + \
                            0.001 * np.dot(diff_root_w, diff_root_w)

        if self.exist_rew_fn_subterm(idx, 'root'):
            diff_root_p = sim_root_p - kin_root_p
            _, diff_root_Q = self._pb_client.getAxisAngleFromQuaternion(
                self._pb_client.getDifferenceQuaternion(sim_root_Q, kin_root_Q))
            diff_root_v = sim_root_v - kin_root_v
            diff_root_w = sim_root_w - kin_root_w
            error['root'] = 1.0 * np.dot(diff_root_p, diff_root_p) + \
                            0.1 * np.dot(diff_root_Q, diff_root_Q) + \
                            0.01 * np.dot(diff_root_v, diff_root_v) + \
                            0.001 * np.dot(diff_root_w, diff_root_w)

        if self.exist_rew_fn_subterm(idx, 'com'):
            diff_com = np.dot(R_sim_f_inv, sim_com-p_sim_f) - np.dot(R_kin_f_inv, kin_com-p_kin_f)
            diff_com_vel = np.dot(R_sim_f_inv, sim_com_vel) - np.dot(R_kin_f_inv, kin_com_vel)
            error['com'] = 1.0 * np.dot(diff_com, diff_com) + \
                           0.1 * np.dot(diff_com_vel, diff_com_vel)

        if self.exist_rew_fn_subterm(idx, 'obj_com'):
            diff_com = np.dot(R_sim_f_inv, obj_sim_com-p_sim_f) - np.dot(R_kin_f_inv, obj_kin_com-p_kin_f)
            diff_com_vel = np.dot(R_sim_f_inv, obj_sim_com_vel) - np.dot(R_kin_f_inv, obj_kin_com_vel)
            error['obj_com'] = 1.0 * np.dot(diff_com, diff_com) + \
                           0.1 * np.dot(diff_com_vel, diff_com_vel)


        if self.exist_rew_fn_subterm(idx,'self_ratio_ig_pos'):
            error['self_ratio_ig_pos'] = 0.0
            ## Reconstruct full matrix
            sim_pairwise_dist_full_mat = []
            kin_pairwise_dist_full_mat = []
            row = pruned_edges[0]
            col = pruned_edges[1]
            ones = np.ones_like(row)
            edges_full_mat = coo_matrix((ones,(row,col)),shape=(len(self._sim_interaction_points[0]),len(self._sim_interaction_points[0]))).toarray()
            for i in range(3):
                sim_pairwise_dist_full_mat_dim = coo_matrix((sim_sim_interaction_graph[:,i],(row,col)),shape=(len(self._sim_interaction_points[0]),len(self._sim_interaction_points[0]))).toarray()
                kin_pairwise_dist_full_mat_dim = coo_matrix((kin_kin_interaction_graph[:,i],(row,col)),shape=(len(self._kin_interaction_points[0]),len(self._kin_interaction_points[0]))).toarray()
                sim_pairwise_dist_full_mat.append(sim_pairwise_dist_full_mat_dim[:,:,np.newaxis])
                kin_pairwise_dist_full_mat.append(kin_pairwise_dist_full_mat_dim[:,:,np.newaxis])
            
            sim_pairwise_dist_full_mat = np.concatenate(sim_pairwise_dist_full_mat,axis=2)
            kin_pairwise_dist_full_mat = np.concatenate(kin_pairwise_dist_full_mat,axis=2)

            non_zeros = np.count_nonzero(edges_full_mat,axis=1)-1 ## Remove the extra self connection
            sim_row_mean_diff = np.sum(sim_pairwise_dist_full_mat,axis=1)/non_zeros[:,np.newaxis]
            kin_row_mean_diff = np.sum(kin_pairwise_dist_full_mat,axis=1)/non_zeros[:,np.newaxis]
            sim_row_mean_diff_mag = np.linalg.norm(sim_row_mean_diff,axis=1)
            kin_row_mean_diff_mag = np.linalg.norm(kin_row_mean_diff,axis=1)


            sim_pairwise_dist_self_ratio = sim_pairwise_dist_full_mat/sim_row_mean_diff_mag[:,np.newaxis]
            kin_pairwise_dist_self_ratio = kin_pairwise_dist_full_mat/kin_row_mean_diff_mag[:,np.newaxis]

            diff = sim_pairwise_dist_self_ratio - kin_pairwise_dist_self_ratio
            dist = np.linalg.norm(diff,axis=2)
            dist = dist * dist
            
            error['self_ratio_ig_pos'] = np.mean(dist)
        if self.exist_rew_fn_subterm(idx, 'self_ratio_ig_vel'):
            error['self_ratio_ig_vel'] = 0.0
            ## Reconstruct full matrix
            sim_pairwise_dist_full_mat = []
            kin_pairwise_dist_full_mat = []
            row = pruned_edges[0]
            col = pruned_edges[1]
            ones = np.ones_like(row)
            edges_full_mat = coo_matrix((ones,(row,col)),shape=(len(self._sim_interaction_points[0]),len(self._sim_interaction_points[0]))).toarray()
            for i in range(3,6,1):
                sim_pairwise_dist_full_mat_dim = coo_matrix((sim_sim_interaction_graph[:,i],(row,col)),shape=(len(self._sim_interaction_points[0]),len(self._sim_interaction_points[0]))).toarray()
                kin_pairwise_dist_full_mat_dim = coo_matrix((kin_kin_interaction_graph[:,i],(row,col)),shape=(len(self._kin_interaction_points[0]),len(self._kin_interaction_points[0]))).toarray()
                sim_pairwise_dist_full_mat.append(sim_pairwise_dist_full_mat_dim[:,:,np.newaxis])
                kin_pairwise_dist_full_mat.append(kin_pairwise_dist_full_mat_dim[:,:,np.newaxis])
            
            sim_pairwise_dist_full_mat = np.concatenate(sim_pairwise_dist_full_mat,axis=2)
            kin_pairwise_dist_full_mat = np.concatenate(kin_pairwise_dist_full_mat,axis=2)

            non_zeros = np.count_nonzero(edges_full_mat,axis=1)-1 ## Remove the extra self connection
            sim_row_mean_diff = np.sum(sim_pairwise_dist_full_mat,axis=1)/non_zeros[:,np.newaxis]
            kin_row_mean_diff = np.sum(kin_pairwise_dist_full_mat,axis=1)/non_zeros[:,np.newaxis]
            sim_row_mean_diff_mag = np.linalg.norm(sim_row_mean_diff,axis=1)
            kin_row_mean_diff_mag = np.linalg.norm(kin_row_mean_diff,axis=1)


            sim_pairwise_dist_self_ratio = sim_pairwise_dist_full_mat/sim_row_mean_diff_mag[:,np.newaxis]
            kin_pairwise_dist_self_ratio = kin_pairwise_dist_full_mat/kin_row_mean_diff_mag[:,np.newaxis]

            diff = sim_pairwise_dist_self_ratio - kin_pairwise_dist_self_ratio
            dist = np.linalg.norm(diff,axis=2)
            dist = dist * dist
            
            error['self_ratio_ig_vel'] = np.mean(dist)

        if self.exist_rew_fn_subterm(idx,'weighted_self_ratio_ig_pos'):
            error['weighted_self_ratio_ig_pos'] = 0.0
            ## Reconstruct full matrix
            sim_pairwise_dist_full_mat = []
            kin_pairwise_dist_full_mat = []
            row = pruned_edges[0]
            col = pruned_edges[1]
            ones = np.ones_like(row)
            kin_dist_weights = np.array(kin_dist_weights)
            edges_full_mat = coo_matrix((ones,(row,col)),shape=(len(self._sim_interaction_points[0]),len(self._sim_interaction_points[0]))).toarray()
            weights_full_mat = coo_matrix((kin_dist_weights,(row,col)),shape=(len(self._sim_interaction_points[0]),len(self._sim_interaction_points[0]))).toarray()

            for i in range(3):
                sim_pairwise_dist_full_mat_dim = coo_matrix((sim_sim_interaction_graph[:,i],(row,col)),shape=(len(self._sim_interaction_points[0]),len(self._sim_interaction_points[0]))).toarray()
                kin_pairwise_dist_full_mat_dim = coo_matrix((kin_kin_interaction_graph[:,i],(row,col)),shape=(len(self._kin_interaction_points[0]),len(self._kin_interaction_points[0]))).toarray()
                sim_pairwise_dist_full_mat.append(sim_pairwise_dist_full_mat_dim[:,:,np.newaxis])
                kin_pairwise_dist_full_mat.append(kin_pairwise_dist_full_mat_dim[:,:,np.newaxis])
            
            sim_pairwise_dist_full_mat = np.concatenate(sim_pairwise_dist_full_mat,axis=2)
            kin_pairwise_dist_full_mat = np.concatenate(kin_pairwise_dist_full_mat,axis=2)

            non_zeros = np.count_nonzero(edges_full_mat,axis=1)#-1 ## Remove the extra self connection
            sim_row_mean_diff = np.sum(sim_pairwise_dist_full_mat,axis=1)/non_zeros[:,np.newaxis]
            kin_row_mean_diff = np.sum(kin_pairwise_dist_full_mat,axis=1)/non_zeros[:,np.newaxis]
            sim_row_mean_diff_mag = np.linalg.norm(sim_row_mean_diff,axis=1)
            kin_row_mean_diff_mag = np.linalg.norm(kin_row_mean_diff,axis=1)


            sim_pairwise_dist_self_ratio = np.nan_to_num(sim_pairwise_dist_full_mat/sim_row_mean_diff_mag[:,np.newaxis])
            kin_pairwise_dist_self_ratio = np.nan_to_num(kin_pairwise_dist_full_mat/kin_row_mean_diff_mag[:,np.newaxis])
            
            diff = sim_pairwise_dist_self_ratio - kin_pairwise_dist_self_ratio
            dist = np.linalg.norm(diff,axis=2)
            dist = dist * dist * weights_full_mat
            
            error['weighted_self_ratio_ig_pos'] = np.sum(dist)
        if self.exist_rew_fn_subterm(idx, 'weighted_self_ratio_ig_vel'):
            error['weighted_self_ratio_ig_vel'] = 0.0
            ## Reconstruct full matrix
            sim_pairwise_dist_full_mat = []
            kin_pairwise_dist_full_mat = []
            row = pruned_edges[0]
            col = pruned_edges[1]
            ones = np.ones_like(row)
            kin_dist_weights = np.array(kin_dist_weights)
            edges_full_mat = coo_matrix((ones,(row,col)),shape=(len(self._sim_interaction_points[0]),len(self._sim_interaction_points[0]))).toarray()
            weights_full_mat = coo_matrix((kin_dist_weights,(row,col)),shape=(len(self._sim_interaction_points[0]),len(self._sim_interaction_points[0]))).toarray()       
            for i in range(3,6,1):
                sim_pairwise_dist_full_mat_dim = coo_matrix((sim_sim_interaction_graph[:,i],(row,col)),shape=(len(self._sim_interaction_points[0]),len(self._sim_interaction_points[0]))).toarray()
                kin_pairwise_dist_full_mat_dim = coo_matrix((kin_kin_interaction_graph[:,i],(row,col)),shape=(len(self._kin_interaction_points[0]),len(self._kin_interaction_points[0]))).toarray()
                sim_pairwise_dist_full_mat.append(sim_pairwise_dist_full_mat_dim[:,:,np.newaxis])
                kin_pairwise_dist_full_mat.append(kin_pairwise_dist_full_mat_dim[:,:,np.newaxis])
            
            sim_pairwise_dist_full_mat = np.concatenate(sim_pairwise_dist_full_mat,axis=2)
            kin_pairwise_dist_full_mat = np.concatenate(kin_pairwise_dist_full_mat,axis=2)

            non_zeros = np.count_nonzero(edges_full_mat,axis=1)-1 ## Remove the extra self connection
            sim_row_mean_diff = np.sum(sim_pairwise_dist_full_mat,axis=1)/non_zeros[:,np.newaxis]
            kin_row_mean_diff = np.sum(kin_pairwise_dist_full_mat,axis=1)/non_zeros[:,np.newaxis]
            sim_row_mean_diff_mag = np.linalg.norm(sim_row_mean_diff,axis=1)
            kin_row_mean_diff_mag = np.linalg.norm(kin_row_mean_diff,axis=1)


            sim_pairwise_dist_self_ratio = sim_pairwise_dist_full_mat/sim_row_mean_diff_mag[:,np.newaxis]
            kin_pairwise_dist_self_ratio = kin_pairwise_dist_full_mat/kin_row_mean_diff_mag[:,np.newaxis]

            diff = sim_pairwise_dist_self_ratio - kin_pairwise_dist_self_ratio
            dist = np.linalg.norm(diff,axis=2)
            dist = dist * dist * weights_full_mat
            
            error['weighted_self_ratio_ig_vel'] = np.sum(dist)
        
        if self.exist_rew_fn_subterm(idx, 'weighted_ig_pos_ratio_distance'):
            error['weighted_ig_pos_ratio_distance'] = 0
            sim_pos_dist =  np.linalg.norm(sim_sim_interaction_graph[:,:3],axis=1)
            kin_pos_dist = np.linalg.norm(kin_kin_interaction_graph[:,:3],axis=1)
            
            ratio = ((sim_pos_dist-kin_pos_dist)/(kin_pos_dist+1e-6))
            ratio = np.clip(ratio,0,5)
            ratio_sq = ratio * ratio * kin_dist_weights
            
            error['weighted_ig_pos_ratio_distance'] = np.sum(ratio_sq)
        if self.exist_rew_fn_subterm(idx, 'weighted_ig_vel_ratio_distance'):
            error['weighted_ig_vel_ratio_distance'] = 0
            sim_vel_dist =  np.linalg.norm(sim_sim_interaction_graph[:,3:],axis=1)
            kin_vel_dist = np.linalg.norm(kin_kin_interaction_graph[:,3:],axis=1)
            
            ratio = ((sim_vel_dist-kin_vel_dist)/(kin_vel_dist+1e-6))
            ratio = np.clip(ratio,0,5)
            ratio_sq = ratio * ratio * kin_dist_weights
            error['weighted_ig_vel_ratio_distance'] = np.sum(ratio_sq)  

        if self.exist_rew_fn_subterm(idx, 'weighted_ig_pos_ratio'):
            error['weighted_ig_pos_ratio'] = 0
            kin_pos_dist = np.linalg.norm(kin_kin_interaction_graph[:,:3],axis=1)
            diff_norm = np.linalg.norm((sim_sim_interaction_graph[:,:3]-kin_kin_interaction_graph[:,:3]),axis=1)

            ratio = (diff_norm/(kin_pos_dist+1e-6))
            ratio = np.clip(ratio,0,5)
            ratio_sq = ratio * ratio * kin_dist_weights
            
            error['weighted_ig_pos_ratio'] = np.sum(ratio_sq) 
        if self.exist_rew_fn_subterm(idx, 'weighted_ig_pos_base_relative'):
            error['weighted_ig_pos_base_relative'] = 0
            ## Reconstruct full matrix
            sim_pairwise_dist_full_mat = []
            kin_pairwise_dist_full_mat = []
            row = pruned_edges[0]
            col = pruned_edges[1]
            ones = np.ones_like(row)
            kin_dist_weights = np.array(kin_dist_weights)
            edges_full_mat = coo_matrix((ones,(row,col)),shape=(len(self._sim_interaction_points[0]),len(self._sim_interaction_points[0]))).toarray()
            weights_full_mat = coo_matrix((kin_dist_weights,(row,col)),shape=(len(self._sim_interaction_points[0]),len(self._sim_interaction_points[0]))).toarray()

            for i in range(3):
                sim_pairwise_dist_full_mat_dim = coo_matrix((sim_sim_interaction_graph[:,i],(row,col)),shape=(len(self._sim_interaction_points[0]),len(self._sim_interaction_points[0]))).toarray()
                kin_pairwise_dist_full_mat_dim = coo_matrix((kin_kin_interaction_graph[:,i],(row,col)),shape=(len(self._kin_interaction_points[0]),len(self._kin_interaction_points[0]))).toarray()
                sim_pairwise_dist_full_mat.append(sim_pairwise_dist_full_mat_dim[:,:,np.newaxis])
                kin_pairwise_dist_full_mat.append(kin_pairwise_dist_full_mat_dim[:,:,np.newaxis])
            
            sim_pairwise_dist_full_mat = np.concatenate(sim_pairwise_dist_full_mat,axis=2)
            kin_pairwise_dist_full_mat = np.concatenate(kin_pairwise_dist_full_mat,axis=2)

            base_sim_full_mat = self._base_sim_ig[idx][:,:,:3]
            base_kin_full_mat = self._base_kin_ig[idx][:,:,:3]

            sim_diff_2_base = (sim_pairwise_dist_full_mat - base_sim_full_mat)*edges_full_mat[:,:,np.newaxis]



            kin_diff_2_base = (kin_pairwise_dist_full_mat - base_kin_full_mat)*edges_full_mat[:,:,np.newaxis]



            sim_diff_2_base_ratio =  np.nan_to_num(sim_diff_2_base / np.linalg.norm(base_sim_full_mat,axis=2)[:,:,np.newaxis])
            kin_diff_2_base_ratio =  np.nan_to_num(kin_diff_2_base / np.linalg.norm(base_kin_full_mat,axis=2)[:,:,np.newaxis])

            self_vert_cnt = self._self_interaction_vert_cnt
            oppo_vert_cnt = self._oppo_interaction_vert_cnt

            sim_diff_2_base_ratio[self_vert_cnt:,:-oppo_vert_cnt] = sim_pairwise_dist_full_mat[self_vert_cnt:,:-oppo_vert_cnt]*edges_full_mat[self_vert_cnt:,:-oppo_vert_cnt,np.newaxis]
            sim_diff_2_base_ratio[:-oppo_vert_cnt,self_vert_cnt:] = sim_pairwise_dist_full_mat[:-oppo_vert_cnt,self_vert_cnt:]*edges_full_mat[:-oppo_vert_cnt,self_vert_cnt:,np.newaxis]

            kin_diff_2_base_ratio[self_vert_cnt:,:-oppo_vert_cnt] = kin_pairwise_dist_full_mat[self_vert_cnt:,:-oppo_vert_cnt]*edges_full_mat[self_vert_cnt:,:-oppo_vert_cnt,np.newaxis]
            kin_diff_2_base_ratio[:-oppo_vert_cnt,self_vert_cnt:] = kin_pairwise_dist_full_mat[:-oppo_vert_cnt,self_vert_cnt:]*edges_full_mat[:-oppo_vert_cnt,self_vert_cnt:,np.newaxis]

            sim_pos_dist = np.linalg.norm(sim_pairwise_dist_full_mat,axis=2)
            kin_pos_dist = np.linalg.norm(kin_pairwise_dist_full_mat,axis=2)

            diff = sim_diff_2_base_ratio - kin_diff_2_base_ratio
            dist_old = np.linalg.norm(diff,axis=2)
            
            dist = np.array(dist_old)
            dist[self_vert_cnt:,:-oppo_vert_cnt] = 0.5 * dist_old[self_vert_cnt:,:-oppo_vert_cnt]/(kin_pos_dist[self_vert_cnt:,:-oppo_vert_cnt]+1e-6) + 0.5* dist_old[self_vert_cnt:,:-oppo_vert_cnt]/(sim_pos_dist[self_vert_cnt:,:-oppo_vert_cnt]+1e-6)
            dist[:-oppo_vert_cnt,self_vert_cnt:] = 0.5 * dist_old[:-oppo_vert_cnt,self_vert_cnt:]/(kin_pos_dist[:-oppo_vert_cnt,self_vert_cnt:]+1e-6) + 0.5* dist_old[:-oppo_vert_cnt,self_vert_cnt:]/(sim_pos_dist[:-oppo_vert_cnt,self_vert_cnt:]+1e-6)
           
            ## Use this to plot the splitted reward and for analysis
            self._full_matrix_dist_raw[idx] = np.array(dist*dist)

            dist = dist * dist * weights_full_mat

            ## Use this to plot the splitted reward and for analysis
            self._full_matrix_dist[idx] = np.array(dist)

            error['weighted_ig_pos_base_relative'] = np.sum(dist)

        if self.exist_rew_fn_subterm(idx, 'weighted_ig_abs_pos_base_relative'):
            error['weighted_ig_abs_pos_base_relative'] = 0
            ## Reconstruct full matrix
            sim_pairwise_dist_full_mat = []
            kin_pairwise_dist_full_mat = []
            row = pruned_edges[0]
            col = pruned_edges[1]
            ones = np.ones_like(row)
            kin_dist_weights = np.array(kin_dist_weights)
            edges_full_mat = coo_matrix((ones,(row,col)),shape=(len(self._sim_interaction_points[0]),len(self._sim_interaction_points[0]))).toarray()
            weights_full_mat = coo_matrix((kin_dist_weights,(row,col)),shape=(len(self._sim_interaction_points[0]),len(self._sim_interaction_points[0]))).toarray()

            for i in range(3):
                sim_pairwise_dist_full_mat_dim = coo_matrix((sim_sim_interaction_graph[:,i],(row,col)),shape=(len(self._sim_interaction_points[0]),len(self._sim_interaction_points[0]))).toarray()
                kin_pairwise_dist_full_mat_dim = coo_matrix((kin_kin_interaction_graph[:,i],(row,col)),shape=(len(self._kin_interaction_points[0]),len(self._kin_interaction_points[0]))).toarray()
                sim_pairwise_dist_full_mat.append(sim_pairwise_dist_full_mat_dim[:,:,np.newaxis])
                kin_pairwise_dist_full_mat.append(kin_pairwise_dist_full_mat_dim[:,:,np.newaxis])
            
            sim_pairwise_dist_full_mat = np.concatenate(sim_pairwise_dist_full_mat,axis=2)
            kin_pairwise_dist_full_mat = np.concatenate(kin_pairwise_dist_full_mat,axis=2)

            base_sim_full_mat = self._base_sim_ig[idx][:,:,:3]
            base_kin_full_mat = self._base_kin_ig[idx][:,:,:3]

            sim_diff_2_base = (sim_pairwise_dist_full_mat - base_sim_full_mat)*edges_full_mat[:,:,np.newaxis]
            
            sim_diff_2_base[14:,:14] = sim_pairwise_dist_full_mat[14:,:14]
            sim_diff_2_base[:14,14:] = sim_pairwise_dist_full_mat[:14,14:]

            kin_diff_2_base = (kin_pairwise_dist_full_mat - base_kin_full_mat)*edges_full_mat[:,:,np.newaxis]
            
            kin_diff_2_base[14:,:14] = kin_pairwise_dist_full_mat[14:,:14]
            kin_diff_2_base[:14,14:] = kin_pairwise_dist_full_mat[:14,14:]

            diff = sim_diff_2_base - kin_diff_2_base
            dist = np.linalg.norm(diff,axis=2)
            dist = dist * dist * weights_full_mat
            error['weighted_ig_abs_pos_base_relative'] = np.sum(dist)

        if self.exist_rew_fn_subterm(idx, 'weighted_ig_pos_ratio_sym_scaled'):
            error['weighted_ig_pos_ratio_sym_scaled'] = 0
            if idx == 0:
                scale = 0.5
            else:
                scale = 1.3
                
            kin_pos_dist = np.linalg.norm(kin_kin_interaction_graph[:,:3]/scale,axis=1)
            sim_pos_dist = np.linalg.norm(sim_sim_interaction_graph[:,:3]/scale,axis=1)

            diff_norm = np.linalg.norm((sim_sim_interaction_graph[:,:3]-kin_kin_interaction_graph[:,:3]),axis=1)

            ratio = 0.5*(diff_norm/(kin_pos_dist+1e-6)) + 0.5*(diff_norm/(sim_pos_dist+1e-6))
            ratio = np.clip(ratio,0,5)
            ratio_sq = ratio * ratio * kin_dist_weights
            
            error['weighted_ig_pos_ratio_sym_scaled'] = np.sum(ratio_sq) 

        if self.exist_rew_fn_subterm(idx, 'weighted_ig_pos_ratio_sym'):
            error['weighted_ig_pos_ratio_sym'] = 0
            kin_pos_dist = np.linalg.norm(kin_kin_interaction_graph[:,:3],axis=1)
            sim_pos_dist = np.linalg.norm(sim_sim_interaction_graph[:,:3],axis=1)

            diff_norm = np.linalg.norm((sim_sim_interaction_graph[:,:3]-kin_kin_interaction_graph[:,:3]),axis=1)

            ratio = 0.5*(diff_norm/(kin_pos_dist+1e-6)) + 0.5*(diff_norm/(sim_pos_dist+1e-6))
            ratio = np.clip(ratio,0,5)
            ratio_sq = ratio * ratio * kin_dist_weights
            
            error['weighted_ig_pos_ratio_sym'] = np.sum(ratio_sq) 
        if self.exist_rew_fn_subterm(idx, 'weighted_ig_vel_ratio'):
            error['weighted_ig_vel_ratio'] = 0
            kin_vel_dist = np.linalg.norm(kin_kin_interaction_graph[:,3:],axis=1)
            diff_norm = np.linalg.norm((sim_sim_interaction_graph[:,3:]-kin_kin_interaction_graph[:,:3:]),axis=1)

            ratio = (diff_norm/(kin_vel_dist+1e-6))
            ratio = np.clip(ratio,0,5)
            ratio_sq = ratio * ratio * kin_dist_weights
            error['weighted_ig_vel_ratio'] = np.sum(ratio_sq)    

        if self.exist_rew_fn_subterm(idx, 'ig_pos_ratio'):
            error['ig_pos_ratio'] = 0
            sim_pos_dist =  np.linalg.norm(sim_sim_interaction_graph[:,:3],axis=1)
            kin_pos_dist = np.linalg.norm(kin_kin_interaction_graph[:,:3],axis=1)
            
            ratio = ((sim_pos_dist-kin_pos_dist)/(kin_pos_dist+1e-6))
            ratio = np.clip(ratio,0,5)
            ratio_sq = ratio * ratio
            
            error['ig_pos_ratio'] = np.mean(ratio_sq)
        if self.exist_rew_fn_subterm(idx, 'ig_vel_ratio'):
            error['ig_vel_ratio'] = 0
            sim_vel_dist =  np.linalg.norm(sim_sim_interaction_graph[:,3:],axis=1)
            kin_vel_dist = np.linalg.norm(kin_kin_interaction_graph[:,3:],axis=1)
            
            ratio = ((sim_vel_dist-kin_vel_dist)/(kin_vel_dist+1e-6))
            ratio = np.clip(ratio,0,5)
            ratio_sq = ratio * ratio
            error['ig_vel_ratio'] = np.mean(ratio_sq)

        if self.exist_rew_fn_subterm(idx, 'im_pos'):
            error['im_pos'] = 0
            assert len(sim_sim_interaction_graph.shape)==2
            diff = np.array(sim_sim_interaction_graph)[:,:3]-np.array(kin_kin_interaction_graph)[:,:3]
            per_joint_dist = np.linalg.norm(diff,axis=1)
            per_joint_dist = per_joint_dist * per_joint_dist
            error['im_pos'] = np.mean(per_joint_dist)
        if self.exist_rew_fn_subterm(idx, 'im_vel'):
            error['im_vel'] = 0
            assert len(sim_sim_interaction_graph.shape)==2
            diff = np.array(sim_sim_interaction_graph)[:,3:]-np.array(kin_kin_interaction_graph)[:,3:]
            per_joint_vel_dist = np.linalg.norm(diff,axis=1)
            per_joint_vel_dist = per_joint_vel_dist * per_joint_vel_dist
            error['im_vel'] = np.mean(per_joint_vel_dist)

        if self.exist_rew_fn_subterm(idx, 'weighted_im_pos'):
            error['weighted_im_pos'] = 0
            assert len(sim_sim_interaction_graph.shape)==2
            diff = np.array(sim_sim_interaction_graph)[:,:3]-np.array(kin_kin_interaction_graph)[:,:3]
            per_joint_dist = np.linalg.norm(diff,axis=1)
            per_joint_dist = per_joint_dist * per_joint_dist * kin_dist_weights

            error['weighted_im_pos'] = np.sum(per_joint_dist)
        if self.exist_rew_fn_subterm(idx, 'weighted_im_vel'):
            error['weighted_im_vel'] = 0
            assert len(sim_sim_interaction_graph.shape)==2
            diff = np.array(sim_sim_interaction_graph)[:,3:]-np.array(kin_kin_interaction_graph)[:,3:]
            per_joint_vel_dist = np.linalg.norm(diff,axis=1)
            per_joint_vel_dist = per_joint_vel_dist * per_joint_vel_dist * kin_dist_weights
            error['weighted_im_vel'] = np.sum(per_joint_vel_dist)

        return error

    def inspect_end_of_episode(self):
        eoe_reason = super().inspect_end_of_episode()
        
        cur_time = self.get_current_time()
        for i in range(self._num_agent):
            name = self._sim_agent[i].get_name()
            if "ref_motion_end" in self._early_term_choices:
                check = cur_time[i] >= self._ref_motion[i].length()
                if check: eoe_reason.append('[%s] end_of_motion'%self._sim_agent[i].get_name())    
            if "root_mismatch_orientation" in self._early_term_choices or \
               "root_mismatch_position" in self._early_term_choices:
                p1, Q1, v1, w1 = self._sim_agent[i].get_root_state()
                p2, Q2, v2, w2 = self._kin_agent[i].get_root_state()
                ''' TODO: remove pybullet Q utils '''
                if "root_mismatch_orientation" in self._early_term_choices:
                    dQ = self._pb_client.getDifferenceQuaternion(Q1, Q2)
                    _, diff = self._pb_client.getAxisAngleFromQuaternion(dQ)
                    # Defalult threshold is 60 degrees
                    thres = self._config['early_term'].get('root_mismatch_orientation_thres', 1.0472)
                    check = diff > thres
                    if check: eoe_reason.append('[%s]:root_mismatch_orientation'%name)
                if "root_mismatch_position" in self._early_term_choices:
                    diff = np.linalg.norm(p2-p1)
                    thres = self._config['early_term'].get('root_mismatch_position_thres', 0.5)
                    check = diff > thres
                    if check: eoe_reason.append('[%s]:root_mismatch_position'%name)
        return eoe_reason        

    def get_ref_motion_time(self):
        cur_time = self.get_current_time()
        if self._repeat_ref_motion:
            for i in range(self._num_agent):
                motion_length = self._ref_motion[i].length()
                while cur_time[i] > motion_length:
                    cur_time[i] -= motion_length
        return cur_time

    def get_current_time(self):
        return self._start_time + self.get_elapsed_time()

    def sample_ref_motion(self, indices=None):
        ref_indices = []
        ref_motions = []
        if indices is None:
            idx = np.random.randint(len(self._ref_motion_all[0]))
        else:
            idx = indices[0]

        for i in range(self._num_agent):
            ref_indices.append(idx)
            ref_motions.append(self._ref_motion_all[i][idx])
        if self._verbose:
            print('Ref. motions selected: ', ref_indices)
        return ref_motions, ref_indices

    def get_phase(self, motion, elapsed_time, mode='linear', **kwargs):
        if mode == 'linear':
            return elapsed_time / motion.length()
        elif mode == 'trigon':
            period = kwargs.get('period', 1.0)
            theta = 2*np.pi * elapsed_time / period
            return np.array([np.cos(theta), np.sin(theta)])
        else:
            raise NotImplementedError

    def check_end_of_motion(self, idx):
        cur_time = self.get_current_time()
        return cur_time[idx] >= self._ref_motion[idx].length()


    def get_render_data(self, idx, agent_type='sim_char'):
        if agent_type == 'sim_char':
            agent = self._sim_agent[idx]
        elif agent_type == 'kin_char':
            agent = self._kin_agent[idx]
        elif agent_type == 'sim_obj':
            agent = self._obj_sim_agent[idx]   
        else:
            agent = self._obj_kin_agent[idx]
        return self._base_env.get_render_data(agent)
    def get_constraints_info(self):
        constraints = []
        for constraint_id in self._constraints:
            if constraint_id is None:
                continue
            constraint_info = self._pb_client.getConstraintInfo(constraint_id)
            child_body_id = constraint_info[2]
            child_link_id = constraint_info[3]
            joint_pos_in_child =constraint_info[7]

            link_state = self._pb_client.getLinkState(child_body_id, child_link_id, computeLinkVelocity=True)
            p,Q = np.array(link_state[0]),np.array(link_state[1])
            
            R = conversions.Q2R(Q)
            constraint_position = R@joint_pos_in_child+p
            constraints.append(constraint_position.tolist())
        return constraints
    def render(self, rm):
        super().render(rm)

        if rm.flag['custom1']:
            for i in range(self._num_agent):
                if rm.flag['sim_model']:
                    sim_interaction_point = self._sim_interaction_points[i]
                    for state in sim_interaction_point:
                        p = state[:3]
                        v = state[3:]
                        rm.gl.glPushMatrix()
                        rm.gl.glTranslatef(p[0],p[1],p[2])
                        rm.gl.glScalef(0.1,0.1,0.1)
                        rm.gl_render.render_sphere(
                            constants.EYE_T, 0.4, color=[1, 0, 0, 1], slice1=16, slice2=16)
                        rm.gl.glPopMatrix()
                        # rm.gl_render.render_arrow(p, p+v, D=0.01, color=[0.5, 0.5, 0.5, 1])

                if rm.flag['kin_model']:
                    kin_interaction_point = self._kin_interaction_points[i]
                    for state in kin_interaction_point:
                        p = state[:3]
                        v = state[3:]
                        rm.gl.glPushMatrix()
                        rm.gl.glTranslatef(p[0],p[1],p[2])
                        rm.gl.glScalef(0.06,0.06,0.06)
                        rm.gl_render.render_sphere(
                            constants.EYE_T, 0.4, color=[1, 0, 0, 1], slice1=16, slice2=16)
                        rm.gl.glPopMatrix()
                        # rm.gl_render.render_arrow(p, p+v, D=0.01, color=[0.5, 0.5, 0.5, 1])
        if rm.flag['custom2']:
            for constraint_id in self._constraints:
                if constraint_id is not None:
                    constraint_info = self._pb_client.getConstraintInfo(constraint_id)
                    child_body_id = constraint_info[2]
                    child_link_id = constraint_info[3]
                    joint_pos_in_child =constraint_info[7]

                    link_state = self._pb_client.getLinkState(child_body_id, child_link_id, computeLinkVelocity=True)
                    p,Q = np.array(link_state[0]),np.array(link_state[1])
                    
                    R = conversions.Q2R(Q)
                    constraint_position = R@joint_pos_in_child+p
                    rm.gl.glPushMatrix()
                    rm.gl.glTranslatef(constraint_position[0],constraint_position[1],constraint_position[2])
                    rm.gl.glScalef(0.14,0.14,0.14)
                    rm.gl_render.render_sphere(
                        constants.EYE_T, 0.4, color=[0, 1, 1, 1], slice1=16, slice2=16)
                    rm.gl.glPopMatrix()

                
        if rm.flag['custom3'] and self.current_interaction:
            
            interaction_mesh = self.current_interaction
            for i in range(self._num_agent):

                edge_index = interaction_mesh[i]
                all_weight = self.compute_distance_weights(i,edge_index,weight_type=self._interaction_weight_type)

                if rm.flag['toggle_interaction'] or rm.flag['sim_model']:
                    sim_interaction_points = self._sim_interaction_points[i]
                    pa = sim_interaction_points[edge_index[0]]
                    pb =  sim_interaction_points[edge_index[1]]
                    for k in range(len(pa)):
                        if self._prune_edges[edge_index[0][k],edge_index[1][k]]==1:
                            weight = min(0.01+all_weight[k]*10,1)
                            color = [0, 0, 1, weight]
                            rm.gl_render.render_line(pa[k], pb[k], color=color,line_width=5)

                if (not rm.flag['toggle_interaction']) or rm.flag['kin_model']:
                        kin_interaction_points = self._kin_interaction_points[i]
                        pa = kin_interaction_points[edge_index[0]]
                        pb =  kin_interaction_points[edge_index[1]]
                        for k in range(len(pa)):
                            weight = min(0.01+all_weight[k]*10,1)
                            color = [1, 0, 0, weight]
                            rm.gl_render.render_line(pa[k], pb[k], color=color,line_width=5)

        if rm.flag['custom5']:
            interaction_mesh = self.current_interaction
            for i in range(self._num_agent):

                edge_index = interaction_mesh[i]
                all_weight = self.compute_distance_weights(i,edge_index,weight_type=self._interaction_weight_type)

                if rm.flag['toggle_interaction'] or rm.flag['sim_model']:
                    sim_interaction_points = self._sim_interaction_points[i]
                    pa = sim_interaction_points[edge_index[0]]
                    pb =  sim_interaction_points[edge_index[1]]
                    for k in range(len(pa)):
                        if self._prune_edges[edge_index[0][k],edge_index[1][k]]==1:
                            weight = min(all_weight[k]*5,1)
                            color_solid = [0, 0, 1, 1]
                            rm.gl_render.render_line(pa[k], pb[k], color=color_solid,line_width=1)

                if (not rm.flag['toggle_interaction']) or rm.flag['kin_model']:
                        kin_interaction_points = self._kin_interaction_points[i]
                        pa = kin_interaction_points[edge_index[0]]
                        pb =  kin_interaction_points[edge_index[1]]
                        for k in range(len(pa)):
                            weight = min(all_weight[k]*5,1)
                            color_solid = [1, 0, 0, 1]
                            rm.gl_render.render_line(pa[k], pb[k], color=color_solid,line_width=1)
                    
        if rm.flag['custom6']:
            import matplotlib.pyplot as plt            
            for idx in range(self._num_agent):
                fig,axes = plt.subplots(nrows=2,ncols=1,figsize=(10, 7))
                weight,dist = self.compute_distance_weights(idx,self.current_interaction[idx],return_dist=True,weight_type=self._interaction_weight_type)

                edge_idx = self.current_interaction[idx]

                labels = []

                for i in range(len(weight)):
                    label_s = self.interaction_point_names[edge_idx[0][i]]
                    label_t = self.interaction_point_names[edge_idx[1][i]]
                    label = str(idx)+"_"+label_s + "_" + label_t
                    labels.append(label)
                        

                top_ten_idx = np.argpartition(weight, -20)[-20:]

                top_weights = weight[top_ten_idx]
                top_labels = np.array(labels)[top_ten_idx]
                top_dists = dist[top_ten_idx]
                sorted_idx = np.flip(top_weights.argsort())
                sorted_label = top_labels[sorted_idx]
                sorted_weight= top_weights[sorted_idx]
                sorted_dist = top_dists[sorted_idx]
                axes[0].bar(sorted_label,sorted_weight)
                axes[0].set_xticklabels(sorted_label,rotation = -30,ha="left",rotation_mode="anchor",fontsize=10)
                axes[0].set_ylabel("Weight")
                axes[0].xaxis.set_visible(False)

                axes[1].bar(sorted_label,sorted_dist)
                axes[1].set_xticklabels(sorted_label,rotation = -30,ha="left",rotation_mode="anchor",fontsize=10)
                axes[1].set_ylabel("Distance")
                fig.tight_layout()
                plt.show()
            
            rm.flag['custom6'] = False
            pass
        if rm.flag['custom7'] and self.current_interaction:
            
            interaction_mesh = self.current_interaction
            if not rm.flag['toggle_agent']:
                agent = 0
            else:
                agent = 1
            if rm.flag['toggle_interaction'] or rm.flag['sim_model']:

                # for i in range(self._num_agent):
                edge_index = interaction_mesh[agent]
                errs = np.array(self._full_matrix_dist[agent])
                sim_interaction_points = self._sim_interaction_points[agent]
                pa = sim_interaction_points[edge_index[0]]
                pb =  sim_interaction_points[edge_index[1]]
                for k in range(len(pa)):
                    if self._prune_edges[edge_index[0][k],edge_index[1][k]]==1:
                        weight = errs[edge_index[0][k],edge_index[1][k]]
                        color = [0, 0, 1, 1000*weight]
                        rm.gl_render.render_line(pa[k], pb[k], color=color,line_width=5)
            if (not rm.flag['toggle_interaction']) or rm.flag['kin_model']:
                edge_index = interaction_mesh[agent]
                errs = np.array(self._full_matrix_dist[agent])
                kin_interaction_points = self._kin_interaction_points[agent]
                pa = kin_interaction_points[edge_index[0]]
                pb =  kin_interaction_points[edge_index[1]]
                for k in range(len(pa)):
                    if self._prune_edges[edge_index[0][k],edge_index[1][k]]==1:
                        weight = errs[edge_index[0][k],edge_index[1][k]]
                        color = [1, 0, 0, 1000*weight]
                        rm.gl_render.render_line(pa[k], pb[k], color=color,line_width=5)
        if rm.flag['sim_model'] and self._include_object:
            rm.gl.glPushAttrib(rm.gl.GL_LIGHTING|rm.gl.GL_DEPTH_TEST|rm.gl.GL_BLEND)
            colors = [rm.COLOR_AGENT]
            for i in range(len(self._obj_sim_agent)):
                sim_agent = self._obj_sim_agent[i]
                rm.bullet_render.render_model(
                    self._pb_client, 
                    sim_agent._body_id,
                    draw_link=True, 
                    draw_link_info=True, 
                    draw_joint=rm.flag['joint'], 
                    draw_joint_geom=True, 
                    link_info_scale=rm.LINK_INFO_SCALE,
                    link_info_line_width=rm.LINK_INFO_LINE_WIDTH,
                    link_info_num_slice=rm.LINK_INFO_NUM_SLICE,
                    color=colors[i])
                p, Q, v, w = sim_agent.get_root_state()
                rm.gl_render.render_arrow(p, p+v, D=0.01, color=[0.5, 0.5, 0.5, 1])
                rm.gl.glPushMatrix()
                rm.gl.glTranslatef(p[0],p[1],p[2])
                rm.gl.glScalef(0.05,0.05,0.05)
                rm.gl_render.render_sphere(
                    constants.EYE_T, 0.4, color=[0, 0, 0, 0.3], slice1=16, slice2=16)
                rm.gl.glPopMatrix()    
            rm.gl.glPopAttrib()
        if rm.flag['kin_model'] and self._include_object:
            rm.gl.glPushAttrib(rm.gl.GL_LIGHTING|rm.gl.GL_DEPTH_TEST|rm.gl.GL_BLEND)
            colors = [rm.COLOR_AGENT]
            for i in range(len(self._obj_sim_agent)):
                agent = self._obj_kin_agent[i]
                rm.bullet_render.render_model(self._pb_client, 
                                                agent._body_id,
                                                draw_link=True,
                                                draw_link_info=False,
                                                draw_joint=rm.flag['joint'],
                                                draw_joint_geom=False, 
                                                color=[1-colors[i][0], 1-colors[i][1], 1-colors[i][2], 0.3])
            
                p, Q, v, w = agent.get_root_state()
                rm.gl_render.render_arrow(p, p+v, D=0.01, color=[0.5, 0.5, 0.5, 1])
                rm.gl.glPushMatrix()
                rm.gl.glTranslatef(p[0],p[1],p[2])
                rm.gl.glScalef(0.05,0.05,0.05)
                rm.gl_render.render_sphere(
                    constants.EYE_T, 0.4, color=[0, 0, 0, 0.3], slice1=16, slice2=16)
                rm.gl.glPopMatrix()
            rm.gl.glPopAttrib()
        if self.env_obj:
            rm.bullet_render.render_model(
                self._pb_client, 
                self.env_obj, 
                draw_link=True, 
                draw_link_info=True, 
                draw_joint=False, 
                draw_joint_geom=False, 
                link_info_line_width=2.0,
                color=[0.6, 0.6, 0.6, 1.0])




if __name__ == '__main__':

    import env_renderer as er
    import render_module as rm
    import argparse
    from fairmotion.viz.utils import TimeChecker

    rm.initialize()
    
    def arg_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', required=True, type=str)
        return parser

    class EnvRenderer(er.EnvRenderer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.time_checker_auto_play = TimeChecker()
            self.reset()
        def reset(self):
            self.env.reset({})
        def one_step(self):
            # a = np.zeros(100)
            self.env.step()
        def extra_render_callback(self):
            if self.rm.flag['follow_cam']:
                p, _, _, _ = env._sim_agent[0].get_root_state()
                self.rm.viewer.update_target_pos(p, ignore_z=True)
            self.env.render(self.rm)
        def extra_idle_callback(self):
            time_elapsed = self.time_checker_auto_play.get_time(restart=False)
            if self.rm.flag['auto_play'] and time_elapsed >= self.env._dt_act:
                self.time_checker_auto_play.begin()
                self.one_step()
        def extra_keyboard_callback(self, key):
            if key == b'r':
                self.reset()
            elif key == b'O':
                size = np.random.uniform(0.1, 0.3, 3)
                p, Q, v, w = self.env._agent[0].get_root_state()
                self.env._obs_manager.throw(p, size=size)
    
    print('=====Humanoid Imitation Environment=====')
    
    args = arg_parser().parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    env = Env(config['config']['env_config'])

    cam = rm.camera.Camera(pos=np.array([12.0, 0.0, 12.0]),
                           origin=np.array([0.0, 0.0, 0.0]), 
                           vup=np.array([0.0, 0.0, 1.0]), 
                           fov=30.0)

    renderer = EnvRenderer(env=env, cam=cam)
    renderer.run()
