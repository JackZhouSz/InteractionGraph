'''
python rllib_driver.py --mode train --spec data/config/train_faircluster_env_imitation_exp30.yaml --project_dir /home/jungdam/Research/ScaDive/ --local_dir XXX
python rllib_driver.py --mode load --spec data/config/train_faircluster_env_imitation_exp30.yaml --project_dir /home/jungdam/Research/ScaDive/ --checkpoint YYY
'''

import os
import sys
import ray

from ray import tune

from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv

from gym import Env as SingleAgentEnv

from collections import deque
import copy

import argparse
import yaml

ip_head = os.getenv('ip_head')
redis_password = os.getenv('redis_password')

print("ip_head:", ip_head)
print("redis_password:", redis_password)

def arg_parser():
    parser = argparse.ArgumentParser()
    ''' Mode for running an experiment '''
    parser.add_argument("--mode", required=True, choices=['train', 'load'])
    ''' Specification file of the expriment '''
    parser.add_argument("--spec", required=True, action='append')
    '''  '''
    parser.add_argument("--checkpoint", action='append', default=[])
    '''  '''
    parser.add_argument("--num_workers", type=int, default=None)
    '''  '''
    parser.add_argument("--num_cpus", type=int, default=1)
    '''  '''
    parser.add_argument("--num_gpus", type=int, default=0)
    '''  '''
    parser.add_argument("--num_envs_per_worker", type=int, default=None)
    '''  '''
    parser.add_argument("--num_cpus_per_worker", type=int, default=None)
    '''  '''
    parser.add_argument("--num_gpus_per_worker", type=int, default=None)
    ''' Directory where the environment and related files are stored '''
    parser.add_argument("--project_dir", type=str, default=None)
    ''' Directory where intermediate results are saved '''
    parser.add_argument("--local_dir", type=str, default=None)
    ''' Verbose '''
    parser.add_argument("--verbose", action='store_true')
    '''  '''
    parser.add_argument("--ip_head", type=str, default=None)
    '''  '''
    parser.add_argument("--password", type=str, default=None)
    '''  '''
    parser.add_argument("--width", type=int, default=1280)
    '''  '''
    parser.add_argument("--height", type=int, default=720)
    '''  '''
    parser.add_argument("--bgcolor", type=str, default="black")
    '''  '''
    parser.add_argument("--renderer", choices=['inhouse', 'bullet_native'], default='inhouse')
    '''  '''
    parser.add_argument("--temp_dir", type=str, default='/tmp/ray/')

    return parser

def is_multiagent(env_cls):
    if MultiAgentEnv in env_cls.__bases__:
        return True
    if SingleAgentEnv in env_cls.__bases__:
        return False
    raise Exception

if __name__ == "__main__":

    args = arg_parser().parse_args()

    assert len(args.spec) > 0

    print('>> Shutdown previous run if exist')
    ray.shutdown()

    if ip_head is not None:
        print('>> Trying to initialize Ray as HEAD...')
        # tmp_dir = os.path.join(spec['local_dir'], os.path.join('tmp/', spec['name']))
        if redis_password:
            ray.init(
                address=ip_head, 
                _redis_password=redis_password, 
                _temp_dir=args.temp_dir)
        else:
            ray.init(address=ip_head, 
                _temp_dir=args.temp_dir)
        print('>> Ray was initialized as HEAD')
    else:
        assert args.num_cpus is not None
        assert args.num_gpus is not None
        print('>> Trying to initialize Ray as CLIENT...')
        print('num_cpus:', args.num_cpus)
        print('num_gpus:', args.num_gpus)
        print('redis_password:', redis_password)
        if redis_password:
            ray.init(
                num_cpus=args.num_cpus, 
                num_gpus=args.num_gpus, 
                _redis_password=redis_password, 
                _temp_dir=args.temp_dir)
        else:
            ray.init(
                num_cpus=args.num_cpus,
                num_gpus=args.num_gpus, 
                _temp_dir=args.temp_dir)
        print('>> Ray was initialized as CLIENT')

    config_list = []
    spec_list = []
    for spec_file in args.spec:
        with open(spec_file) as f:
            spec = yaml.load(f, Loader=yaml.FullLoader)
        config = spec['config']

        '''
        Register environment to learn according to the input specification file
        '''

        if config['env'] == "HumanoidFight":
            from envs import rllib_env_fight as env_module
        elif config['env'] == "HumanoidChase":
            from envs import rllib_env_chase as env_module
        elif config['env'] == "HumanoidImitation":
            from envs import rllib_env_imitation as env_module
        elif config['env'] == "HumanoidImitationContrastive":
            from envs import rllib_env_imitation_contrastive as env_module
        elif config['env'] == "HumanoidFencing":
            from envs import rllib_env_fencing as env_module
        elif config['env'] == "HumanoidFencingHierarchy":
            from envs import rllib_env_fencing_hierarchy as env_module
        elif config['env'] == "HumanoidBoxing":
            from envs import rllib_env_boxing as env_module
        elif config['env'] == "HumanoidBoxingHierarchy":
            from envs import rllib_env_boxing_hierarchy as env_module
        elif config['env'] == "HumanoidBoxingHierarchyMOE":
            from envs import rllib_env_boxing_hierarchy_moe as env_module
        elif config['env'] == "HumanoidBoxingSolo":
            from envs import rllib_env_boxing_solo as env_module
        elif config['env'] == "HumanoidKinematicsContrastive":
            from envs import rllib_env_kinematics_contrastive as env_module
        elif config['env'] == "Contrastive":
            from envs import rllib_env_contrastive as env_module
        elif config['env'] == "HumanoidFollower":
            from envs import rllib_env_follower as env_module
        elif config['env'] == "HumanoidTag":
            from envs import rllib_env_tag as env_module
        elif config['env'] == "HumanoidCrowd":
            from envs import rllib_env_crowd as env_module
        elif config['env'] == "MultiAgentTestEnv":
            from envs import rllib_env_test as env_module
        elif config['env'] == "HumanoidHighFive":
            from envs import rllib_env_highfive as env_module
        elif config['env'] == "HumanoidImitationRetarget":
            from envs import rllib_env_imitation_retarget as env_module
        elif config['env'] == "HumanoidImitationInteraction":
            from envs import rllib_env_imitation_interaction as env_module       
        elif config['env'] == "HumanoidImitationInteractionGraph":
            from envs import rllib_env_imitation_interaction_graph as env_module
        elif config['env'] == 'HumanoidImitationInteractionGraphTwo':
            from envs import rllib_env_imitation_interaction_graph_two as env_module
        elif config['env'] == "HumanoidSitdown":
            from envs import rllib_env_sitdown as env_module
        else:
            raise NotImplementedError("Unknown Environment")

        # register_env(config['env'], lambda config: env_module.env_cls(config))
        config['env_config']['horizon'] = config['horizon']
        config['env'] = env_module.env_cls
        '''
        Register custom model to use if it exists
        '''

        framework = config.get('framework')

        if config.get('model'):
            custom_model = config.get('model').get('custom_model')
            if custom_model:
                if framework=="torch":
                    import rllib_model_torch
                else:
                    raise NotImplementedError

        '''
        Validate configurations and overide values by arguments
        '''

        if args.local_dir is not None:
            spec.update({'local_dir': args.local_dir})
        
        if args.project_dir is not None:
            assert os.path.exists(args.project_dir)
            config['env_config']['project_dir'] = args.project_dir
            if 'base_env_config' in config['env_config']:
                config['env_config']['base_env_config']['project_dir'] =\
                    args.project_dir
        
        if config['model'].get('custom_model_config'):
            config['model']['custom_model_config'].update(
                {'project_dir': config['env_config']['project_dir']})

        if args.verbose:
            config['env_config'].update({'verbose': args.verbose})
            if 'base_env_config' in config['env_config']:
                config['env_config']['base_env_config'].update(
                    {'verbose': args.verbose})
        
        if args.num_workers is not None:
            config.update({'num_workers': args.num_workers})
        
        if args.num_gpus is not None:
            config.update({'num_gpus': args.num_gpus})

        if args.num_envs_per_worker:
            config.update({'num_envs_per_worker': args.num_envs_per_worker})

        if args.num_cpus_per_worker:
            config.update({'num_cpus_per_worker': args.num_cpus_per_worker})

        if args.num_gpus_per_worker:
            config.update({'num_gpus_per_worker': args.num_gpus_per_worker})

        if args.mode == "train":
            if not os.path.exists(spec['local_dir']):
                raise Exception(
                    "The directory does not exist: %s"%spec['local_dir'])

        config_override = env_module.config_override(spec)
        config.update(config_override)
        def adjust_config(config, alg):
            rollout_fragment_length = config.get('rollout_fragment_length')
            num_workers = config.get('num_workers')
            num_envs_per_worker = config.get('num_envs_per_worker')
            train_batch_size = config.get('train_batch_size')

            ''' 
            Set rollout_fragment_length value so that
            workers can genertate train_batch_size tuples correctly
            '''
            rollout_fragment_length = \
                max(train_batch_size // (num_workers * num_envs_per_worker), 1)
        
            # while rollout_fragment_length * num_workers * num_envs_per_worker \
            #     < train_batch_size:
            #     rollout_fragment_length += 1

            config['rollout_fragment_length'] = rollout_fragment_length

            if alg in ["APPO"]:
                config['train_batch_size'] = config['sgd_minibatch_size']
                config.pop('sgd_minibatch_size')
                if config.get('vf_clip_param') is not None:
                    config.pop('vf_clip_param')
                if config.get('vf_share_layers') is not None:
                    config.pop('vf_share_layers')
            
            if alg in ['DDPPO', 'DDPPO_CUSTOM']:
                # print('rollout_fragment_length', rollout_fragment_length)
                config['train_batch_size'] = -1
                ## This is a temporary fix for RLlib (1.6.0)...
                # config['rollout_fragment_length'] = train_batch_size // num_envs_per_worker
                # config['sgd_minibatch_size'] = min(
                #     max(config['sgd_minibatch_size'] // num_workers, 50),
                #     rollout_fragment_length)

        adjust_config(config, spec['run'])

        spec_list.append(spec)
        config_list.append(config)
    
    if args.mode == "load":

        def adjust_config_for_loading(config):
            config["num_workers"] = 1
            config['num_envs_per_worker'] = 1
            config['num_cpus_per_worker'] = 1
            config['num_gpus_per_worker'] = 0
            config['remote_worker_envs'] = False
            if config.get('callbacks') is not None:
                config['callbacks'] = None

        def load_trainer_cls(spec):
            if spec["run"] == "PPO":
                from ray.rllib.agents.ppo import PPOTrainer as Trainer
            elif spec["run"] == "APPO":
                from ray.rllib.agents.ppo import APPOTrainer as Trainer
            elif spec["run"] == "PPO_CUSTOM":
                from rllib_ppo_custom import Trainer as Trainer
            elif spec["run"] == "DDPPO":
                from ray.rllib.agents.ppo import DDPPOTrainer as Trainer
            else:
                raise NotImplementedError
            return Trainer

        adjust_config_for_loading(config_list[0])

        trainers = []
        
        trainer_cls = load_trainer_cls(spec_list[0])
        trainer = trainer_cls(env=env_module.env_cls, config=config_list[0])
        if len(args.checkpoint) > 0:
            trainer.restore(args.checkpoint[0])
        trainers.append(trainer)

        if len(args.spec)==2 and len(args.checkpoint)==2:
            adjust_config_for_loading(config_list[1])
            trainer_cls = load_trainer_cls(spec_list[1])
            trainer = trainer_cls(env=env_module.env_cls, config=config_list[1])
            trainer.restore(args.checkpoint[1])
            trainers.append(trainer)

        env_module.rm.initialize()

        env = env_module.env_cls(config_list[0]['env_config'])
        cam = env_module.default_cam(env)

        if args.bgcolor == "black":
            bgcolor = [0.0, 0.0, 0.0, 1.0]
        elif args.bgcolor == "white":
            bgcolor = [1.0, 1.0, 1.0, 1.0]
        elif args.bgcolor == "clear":
            bgcolor = [0.0, 0.0, 0.0, 0.0]
        else:
            raise NotImplementedError

        renderer = env_module.EnvRenderer(
            trainers=trainers, 
            env=env, 
            cam=cam, 
            renderer=args.renderer, 
            size=(args.width, args.height), 
            bgcolor=bgcolor, 
            config=config_list[0])
        renderer.run()
    else:
        spec = spec_list[0]
        config = config_list[0]
        resume = False
        if len(args.checkpoint) > 0:
            checkpoint = args.checkpoint[0]
        else:
            checkpoint = None
        if checkpoint is None and os.path.exists(
            os.path.join(spec['local_dir'], spec['name'])
        ):
            resume = "ERRORED_ONLY"
        if spec['run'] in ["PPO", "PPO_CUSTOM", "APPO", "DDPPO", "DDPPO_CUSTOM"]:
            callbacks = None
            if spec['run'] == "DDPPO_CUSTOM":
                import rllib_ddppo_custom
            if config.get('callbacks') == "InitialStateDistributionCallback":
                from callbacks.initial_state_distrubution_callback import InitialStateDistributionCallback
                config['callbacks']=InitialStateDistributionCallback
                print("callback added to config")
                # callbacks = InitialStateDistributionCallback
            gettrace = getattr(sys, 'gettrace',None)
            if gettrace():
                def load_trainer_cls(spec):
                    if spec["run"] == "PPO":
                        from ray.rllib.agents.ppo import PPOTrainer as Trainer
                    elif spec["run"] == "APPO":
                        from ray.rllib.agents.ppo import APPOTrainer as Trainer
                    elif spec["run"] == "PPO_CUSTOM":
                        from rllib_ppo_custom import Trainer as Trainer
                    elif spec["run"] == "DDPPO":
                        from ray.rllib.agents.ppo import DDPPOTrainer as Trainer
                    else:
                        raise NotImplementedError
                    return Trainer

                trainer_cls = load_trainer_cls(spec)
                trainer = trainer_cls(env=env_module.env_cls, config=config)
                for _ in range(20):
                    trainer.train()
            else: 
                tune.run(
                    spec['run'],
                    name=spec['name'],
                    stop=spec['stop'],
                    local_dir=spec['local_dir'],
                    checkpoint_freq=spec['checkpoint_freq'],
                    checkpoint_at_end=spec['checkpoint_at_end'],
                    config=config,
                    resume=resume,
                    # callbacks=callbacks,
                    restore=checkpoint,
                    reuse_actors=spec.get('reuse_actors', True),
                    raise_on_failed_trial=False,
                )
        elif spec['run'] == "PPO_MULTIAGENT_CUSTOM1":
            from fairmotion.viz.utils import TimeChecker

            config1_override, config2_override = env_module.config_override_custom(spec)
            config1 = copy.deepcopy(config)
            config2 = copy.deepcopy(config)
            config1.update(config1_override)
            config2.update(config1_override)

            trainer_info_list = [
                {
                    'trainer': Trainer(env=env_module.env_cls, config=config),
                    'weight_buffer': deque(maxlen=10),
                    'weight_last': None,
                },
                {
                    'trainer': Trainer(env=env_module.env_cls, config=config),
                    'weight_buffer': deque(maxlen=10),
                    'weight_last': None,
                },
            ]

            def save_checkpoint(trainer_info_list):
                for i, trainer_info in enumerate(trainer_info_list):
                    checkpoint = trainer_info['trainer'].save()
                    print("[%d]: checkpoint saved at %s"%(i+1, checkpoint))

            if args.checkpoint is not None:
                trainer_info_list[0]['trainer'].resotre(args.checkpoint)
                print("[1]: checkpoint loaded from %s"%(args.checkpoint))
            if args.checkpoint2 is not None:
                trainer_info_list[1]['trainer'].resotre(args.checkpoint2)
                print("[2]: checkpoint loaded from %s"%(args.checkpoint2))

            iteration = 1
            trainer_change_num = 0
            trainer_change_freq = 10
            
            checkpoint_freq = spec.get('checkpoint_freq')
            tc = TimeChecker()

            for trainer_info in trainer_info_list:
                trainer_info['weight_last'] = trainer_info['trainer'].get_weights()
                trainer_info['weight_buffer'].append(trainer_info['trainer'].get_weights())

            while True:
                ''' Check conditions for termination '''
                finished = False
                time_total_s = spec['stop'].get('time_total_s')
                if time_total_s:
                    finished = tc.get_time(restart=False) >= time_total_s
                if finished:
                    if spec.get('checkpoint_at_end'):
                        save_checkpoint(trainer_info_list)
                    break
                
                ''' Save '''
                if checkpoint_freq and iteration%checkpoint_freq==0:
                    save_checkpoint(trainer_info_list)

                trainer_info1 = trainer_info_list[trainer_change_num%2]
                trainer_info2 = trainer_info_list[(trainer_change_num+1)%2]

                ''' Sample an opponent policy ''' 
                trainer_info2['trainer'].set_weights(
                    random.choice(trainer_info2['weight_buffer']))

                ''' Train ''' 
                result = trainer_info1['trainer'].train()
                print(pretty_print(result))

                if iteration % trainer_change_freq == 0:
                    trainer_info1['weight_last'] = trainer_info1.get_weights()
                    trainer_info2['trainer'].set_weights(trainer_info2['weight_last'])

                    trainer_info1['weight_buffer'].append(trainer_info1['trainer'].get_weights())
                    trainer_info2['weight_buffer'].append(trainer_info2['trainer'].get_weights())
                    
                    trainer_change_num += 1

                iteration += 1
