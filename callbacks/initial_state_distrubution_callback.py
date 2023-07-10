import pdb
import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks
from typing import Dict, Optional, TYPE_CHECKING
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.typing import AgentID, PolicyID
import ray
import matplotlib.pyplot as plt
import pickle as pkl
import os
class InitialStateDistributionCallback(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super().__init__(legacy_callbacks_dict)

        self.initial_state_bins = None
        self.initial_state_count = None
    
    def on_episode_start(self, 
                         *,
                         worker: "RolloutWorker",
                         base_env: BaseEnv,
                         policies: Dict[PolicyID, Policy],
                         episode: MultiAgentEpisode,
                         env_index: Optional[int] = None,
                         **kwargs) -> None:
        episode.hist_data['episode_expected_duration'] = []
        episode.hist_data['episode_duration'] = []
        episode.hist_data['episode_start_time'] = []
        episode.hist_data['ratio'] = []
    def on_episode_end(self,
                       *,
                       worker: "RolloutWorker",
                       base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode,
                       env_index: Optional[int] = None,
                       **kwargs) -> None:
                       
        end_time = episode.last_info_for('player1')['episode_current_time']
        start_time = episode.last_info_for('player1')['episode_start_time']
        expected_duration = episode.last_info_for('player1')['episode_expected_duration']

        ## Fix the episode length
        ratio = (end_time-start_time)/expected_duration
        full_length = episode.last_info_for('player1')['episode_full_length']
        episode.hist_data['episode_duration'] = [(end_time-start_time)]
        episode.hist_data['episode_start_time'] = [start_time]
        episode.hist_data['episode_expected_duration'] = [expected_duration]
        episode.hist_data['ratio'] = [ratio]
        episode.hist_data['full_length']=[full_length]
        # episode.hist_data['episode_duration'] = [0]
        # episode.hist_data['episode_start_time'] = [0]
        # episode.hist_data['episode_expected_duration'] = [0]
        # episode.hist_data['ratio'] = [0]
        # episode.hist_data['full_length']=[0]

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        # Called every iteration
        print("On Training Result")
        if result['hist_stats'].get('full_length',None) is None:
            print("Why is hist stats empty?")
            return
        config = result['config']
        fps = config['env_config']['fps_con']
        iteration = trainer.iteration
        checkpoint_freq = 10
        if self.initial_state_bins is None:
            full_length = result['hist_stats']['full_length'][-1]
            full_length_frames = int(full_length * fps)
            self.initial_state_bins = np.zeros(full_length_frames)
            self.initial_state_count = np.zeros(full_length_frames)

        start_times = result['hist_stats']['episode_start_time']
        start_times_frames = (np.array(start_times)*fps).astype(int)
        ratios = result['hist_stats']['ratio']
        for i in range(len(start_times_frames)):
            ratio = ratios[i]
            frame = start_times_frames[i]
            self.initial_state_bins[frame] += ratio
            self.initial_state_count[frame] +=1

        average_states = np.clip(np.nan_to_num(self.initial_state_bins/self.initial_state_count),0,1)
        average_ratio_inverse_dist = 1-average_states
        average_ratio_inverse_dist = average_ratio_inverse_dist/np.sum(average_ratio_inverse_dist)
        
        def set_init_dist(env):
            env.set_task.remote(average_ratio_inverse_dist)

        # ray.util.pdb.set_trace()
        trainer.workers.foreach_env(
                lambda env: set_init_dist(env)
        )
        # ray.get(all_collected)

        if iteration % checkpoint_freq == 0 or iteration == 1:
            new_dist_data = {

                'total_ratios':self.initial_state_bins,
                'total_counts':self.initial_state_count,
                'average_ratios':average_states,
                'average_ratio_inverse_dist':average_ratio_inverse_dist,
            }

            logdir = trainer.logdir
            save_dir = os.path.join(logdir,"init_dist_data_%05d.pkl"%iteration)
            print("Save Dir: ",save_dir)
            with open(save_dir,'wb') as f:
                pkl.dump(new_dist_data,f)
