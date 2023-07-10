import logging
from math import floor
from black import out
import numpy as np
import os

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, SlimConv2d,normc_initializer
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.annotations import override
# from ray.rllib.utils.framework import get_activation_fn
from ray.rllib.utils import try_import_torch
from ray.rllib.models import ModelCatalog
torch, nn = try_import_torch()
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def softmax_normalized(x, dim):
    x_hat = x-torch.max(x, dim=dim)[0].unsqueeze(-1)
    return F.softmax(x_hat, dim=dim)

def get_activation_fn(name=None):
    
    if name in ["linear", None]:
        return None
    if name in ["swish", "silu"]:
        from ray.rllib.utils.torch_ops import Swish
        return Swish
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    if name == "sigmoid":
        return nn.Sigmoid
    if name == "elu":
        return nn.ELU

    raise ValueError("Unknown activation ({})={}!".format(name))

def create_layer(layer_type, layers, size_in, size_out, append_log_std=False):
    output_layer = None
    lstm_layer = None
    if layer_type == "mlp":
        param = {
            "size_in": size_in, 
            "size_out": size_out, 
            "layers": layers,
            "append_log_std": append_log_std,
        }
        output_layer = FC(**param)
    elif layer_type == "lstm":
        ## TODO: needed to be fixed by layers
        assert layers[0]["type"] == "lstm"
        hidden_size = layers[0]["hidden_size"]
        num_layers = layers[0]["num_layers"]
        lstm_layer = nn.LSTM(
            input_size=size_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        ''' For the compatibility of the previous config '''
        output_activation = layers[0].get("output_activation")
        if output_activation:
            if output_activation == "linear":
                output_layer = nn.Sequential(
                    nn.Linear(
                        in_features=hidden_size,
                        out_features=size_out,
                    ),
                )
            elif output_activation == "tanh":
                output_layer = nn.Sequential(
                    nn.Linear(
                        in_features=hidden_size,
                        out_features=size_out,
                    ),
                    nn.Tanh(),
                )
            else:
                raise NotImplementedError

        output_layers = layers[0].get("output_layers")
        if output_layers:
            param = {
                "size_in": hidden_size, 
                "size_out": size_out, 
                "layers": output_layers,
                "append_log_std": append_log_std,
            }
            output_layer = FC(**param)
    else:
        raise NotImplementedError
    assert output_layer is not None
    return output_layer, lstm_layer

def forward_layer(obs, seq_lens=None, state=None, state_cnt=None, output_layer=None, lstm_layer=None):
    if lstm_layer is None and output_layer is None:
        out = obs
    elif lstm_layer is None and output_layer is not None:
        out = output_layer(obs)
    elif lstm_layer is not None and output_layer is not None:
        assert seq_lens is not None and state is not None and state_cnt is not None
        out, state_cnt = process_lstm(obs, seq_lens, state, state_cnt, output_layer, lstm_layer)
    else:
        raise Exception("Invalid Inputs")
    return out, state_cnt

def process_lstm(obs, seq_lens, state, state_cnt, output_layer, lstm_layer):
    if isinstance(seq_lens, np.ndarray):
        seq_lens = torch.Tensor(seq_lens).int()
    if seq_lens is not None:
        max_seq_len = obs.shape[0] // seq_lens.shape[0]
        # max_seq_len=torch.max(seq_lens)
    
    input_lstm = add_time_dimension(
        obs,
        max_seq_len=max_seq_len,
        framework="torch",
        time_major=False,
    )

    ''' 
    Assume that the shape of state is 
    (batch, num_layers * num_directions, hidden_size). So we change
    the first axis with the second axis.
    '''
    h_lstm, c_lstm = state[state_cnt], state[state_cnt+1]

    h_lstm = h_lstm.reshape(h_lstm.shape[1], h_lstm.shape[0], h_lstm.shape[2])
    c_lstm = c_lstm.reshape(c_lstm.shape[1], c_lstm.shape[0], c_lstm.shape[2])
    
    output_lstm, (h_lstm, c_lstm) = lstm_layer(input_lstm, (h_lstm, c_lstm))
    output_lstm = output_lstm.reshape(-1, output_lstm.shape[-1])
    out = output_layer(output_lstm)

    '''
    Change the first and second axes of the output state so that
    it matches to the assumption
    '''
    h_lstm = h_lstm.reshape(h_lstm.shape[1], h_lstm.shape[0], h_lstm.shape[2])
    c_lstm = c_lstm.reshape(c_lstm.shape[1], c_lstm.shape[0], c_lstm.shape[2])

    state[state_cnt] = h_lstm
    state[state_cnt+1] = c_lstm

    state_cnt += 2

    return out, state_cnt

class Normalizer(nn.Module):
    def __init__(self, mu, std):
        super().__init__()
        self.set_val(mu, std)

    def set_val(self, mu, std):
        assert mu.shape[-1] == std.shape[-1]
        self.mu = mu
        self.std = std
    
    def forward(self, x):
        assert x.shape[-1] == self.mu.shape[-1]
        return (x - self.mu) / self.std

class Denormalizer(nn.Module):
    def __init__(self, mu, std):
        super().__init__()
        self.set_val(mu, std)

    def set_val(self, mu, std):
        assert mu.shape[-1] == std.shape[-1]
        self.mu = mu
        self.std = std
    
    def forward(self, x):
        assert x.shape[-1] == self.mu.shape[-1]
        return (x + self.mu) * self.std


class AppendLogStd(nn.Module):
    '''
    An appending layer for log_std.
    '''
    def __init__(self, type, init_val, dim):
        super().__init__()
        self.type = type

        if np.isscalar(init_val):
            init_val = init_val * np.ones(dim)
        elif isinstance(init_val, (np.ndarray, list)):
            assert len(init_val) == dim
        else:
            raise NotImplementedError

        self.init_val = init_val

        if self.type=="constant":
            self.log_std = torch.Tensor(init_val)
        elif self.type=="state_independent":
            self.log_std = torch.nn.Parameter(
                torch.Tensor(init_val))
            self.register_parameter("log_std", self.log_std)
        else:
            raise NotImplementedError

    def set_val(self, val):
        assert self.type=="constant", \
            "Change value is only allowed in constant logstd"
        assert np.isscalar(val), \
            "Only scalar is currently supported"

        self.log_std[:] = val
    
    def forward(self, x):
        assert x.shape[-1] == self.log_std.shape[-1]
        
        shape = list(x.shape)
        for i in range(0, len(shape)-1):
            shape[i] = 1
        log_std = torch.reshape(self.log_std, shape)
        shape = list(x.shape)
        shape[-1] = 1
        log_std = log_std.repeat(shape)

        out = torch.cat([x, log_std], axis=-1)
        return out

class Hardmax(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
    def forward(self, x):
        idx = torch.argmax(x, dim=1)
        # print(x.shape)
        # print(idx, self.num_classes)
        y = F.one_hot(idx, num_classes=self.num_classes)
        # print(y)
        return y

def get_initializer(info):
    if info['name'] == "normc":
        return normc_initializer(info['std'])
    elif info['name'] == 'xavier_normal':
        def initializer(tensor):
            return nn.init.xavier_normal_(tensor, gain=info['gain'])
        return initializer
    elif info['name'] == 'xavier_uniform':
        def initializer(tensor):
            return nn.init.xavier_uniform_(tensor, gain=info['gain'])
        return initializer
    else:
        raise NotImplementedError


class FC(nn.Module):
    ''' 
    A network with fully connected layers.
    '''
    def __init__(self, size_in, size_out, layers, append_log_std=False,
                 log_std_type='constant', sample_std=1.0):
        super().__init__()
        nn_layers = []
        prev_layer_size = size_in
        for l in layers:
            layer_type = l['type']
            if layer_type == 'fc':
                assert isinstance(l['hidden_size'] , int) or l['hidden_size'] =='output'
                hidden_size = l['hidden_size'] if l['hidden_size'] != 'output' else size_out
                layer = SlimFC(
                    in_size=prev_layer_size,
                    out_size=hidden_size,
                    initializer=get_initializer(l['init_weight']),
                    activation_fn=get_activation_fn(l['activation'])
                )
                prev_layer_size = hidden_size
            elif layer_type in ['bn', 'batch_norm']:
                layer = nn.BatchNorm1d(prev_layer_size)
            elif layer_type in ['sm', 'softmax']:
                layer = nn.Softmax(dim=1)
            elif layer_type in ['hm', 'hardmax']:
                layer = Hardmax(num_classes=prev_layer_size)
            else:
                raise NotImplementedError(
                    "Unknown Layer Type:", layer_type)
            nn_layers.append(layer)

        if append_log_std:
            nn_layers.append(AppendLogStd(
                type=log_std_type, 
                init_val=np.log(sample_std), 
                dim=size_out))

        self._model = nn.Sequential(*nn_layers)
    
    def forward(self, x):
        return self._model(x)

    def save_weights(self, file):
        torch.save(self.state_dict(), file)

    def load_weights(self, file):
        self.load_state_dict(torch.load(file))
        self.eval()

    ''' 
    A policy that generates action and value with FCNN
    '''
    DEFAULT_CONFIG = {
        "log_std_type": "constant",
        "sample_std": 1.0,
        "interaction_out_channel":3,
        "interaction_fc_out_shape":128,
        "interaction_net_type":"conv",
        "interaction_layers":[
            {"type": "conv","channel_out":10,"kernel":5,"stride":2,"padding":2,"activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "mp", "kernel":3,"stride":1,"padding":0},
            {"type": "conv","channel_out":"output","kernel":3,"stride":1,"padding":1,"activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
        ],
        "interaction_fc_layers":[
            {"type": "fc", "hidden_size": "out", "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
        ],


        "policy_fn_layers": [
            {"type": "fc", "hidden_size": 256, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": 256, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": "output", "activation": "linear", "init_weight": {"name": "normc", "std": 0.01}},
        ],
        
        "log_std_fn_hiddens": [64, 64],
        "log_std_fn_activations": ["relu", "relu", "linear"],
        "log_std_fn_init_weights": [1.0, 1.0, 0.01],
        "log_std_fn_base": 0.0,
        
        "value_fn_layers": [
            {"type": "fc", "hidden_size": 256, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": 256, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": "output", "activation": "linear", "init_weight": {"name": "normc", "std": 0.01}},
        ],
        "interaction_obs_dim" : None,
        "interaction_obs_num" : None,
        "interaction_feature_dim": None,
    }
    """Generic fully connected network."""
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **model_kwargs):

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        ''' Load and check configuarations '''

        custom_model_config = InteractionPolicy.DEFAULT_CONFIG.copy()
        custom_model_config_by_user = model_config.get("custom_model_config")
        if custom_model_config_by_user:
            custom_model_config.update(custom_model_config_by_user)

        '''
        constant
            log_std will not change during the training
        state_independent
            log_std will be learned during the training
            but it does not depend on the state of the agent
        state_dependent:
            log_std will be learned during the training
            and it depens on the state of the agent
        '''

        log_std_type = custom_model_config.get("log_std_type")
        assert log_std_type in \
            ["constant", "state_independent", "state_dependent"]

        sample_std = custom_model_config.get("sample_std")
        assert np.array(sample_std).all() > 0.0, \
            "The value shoulde be positive"

        assert num_outputs % 2 == 0, (
            "num_outputs must be divisible by two", num_outputs)
        num_outputs = num_outputs//2
        append_log_std = (log_std_type != "state_dependent")

        policy_fn_layers = custom_model_config.get("policy_fn_layers")
        
        value_fn_layers = custom_model_config.get("value_fn_layers")

        interaction_layers = custom_model_config.get("interaction_layers")
        interaction_out_channel = custom_model_config.get("interaction_out_channel")
        interaction_fc_out_shape = custom_model_config.get("interaction_fc_out_shape")
        interaction_fc_layers = custom_model_config.get("interaction_fc_layers")
        self.interaction_net_type = custom_model_config.get("interaction_net_type")
        self._interaction_obs_dim = custom_model_config.get("interaction_obs_dim")
        self._interaction_obs_num = custom_model_config.get("interaction_obs_num")
        self._interaction_feature_dim = custom_model_config.get("interaction_feature_dim")
        self._sparse_interaction = custom_model_config.get("sparse_interaction")

        dim_state = int(np.product(obs_space.shape))
        self._dim_fc = dim_state - self._interaction_obs_dim*self._interaction_obs_num

        print("Interaction Network Dimension: \nFC Input: %d, \nInteraction Input Dim: %d \nInteraction Input Num %d"%(self._dim_fc,self._interaction_obs_dim,self._interaction_obs_num))
        if self.interaction_net_type=="conv":

            def compute_conv_out_flat_dim(dim_in,layers):
                new_dim = dim_in

                for l in layers:
                    if l['type']=='conv' or l['type']=='mp':
                        k = l['kernel']
                        s = l['stride']
                        p = l['padding']
                        new_dim = floor((new_dim-k+2*p)/s)+1
                    else:
                        continue
                return new_dim*new_dim*interaction_out_channel 
            
            def conv2d_fc_fn(params):
                
                out_shape = compute_conv_out_flat_dim(self._interaction_feature_dim[0],params['layers'])
                conv_fn = Conv2D(**params)

                l = interaction_fc_layers[0]
                fc_fn = SlimFC(
                    in_size=out_shape,
                    out_size=interaction_fc_out_shape,
                    initializer=get_initializer(l['init_weight']),
                    activation_fn=get_activation_fn(l['activation'])
                )
                
                out_fn = nn.Sequential(conv_fn,fc_fn)
                return out_fn

            conv_params = {
                "channel_in":3,
                "channel_out":interaction_out_channel,
                "layers":interaction_layers,
            }

            self._pos_interaction_net = conv2d_fc_fn(conv_params)
            self._vel_interaction_net = conv2d_fc_fn(conv_params)

            self._pos_interaction_net_vf = conv2d_fc_fn(conv_params)
            self._vel_interaction_net_vf = conv2d_fc_fn(conv_params)
            downstream_in_size =  self._dim_fc+self._interaction_obs_num*interaction_fc_out_shape*2 + self._interaction_obs_num *self._interaction_feature_dim[0]*self._interaction_feature_dim[1]
        elif self.interaction_net_type == "gcn":
            num_nodes_per_batch = self._interaction_feature_dim[0]

            gnn_params = {
                "channel_in":6,
                "channel_out":interaction_out_channel,
                "out_dim": interaction_fc_out_shape,
                "num_nodes_per_batch":num_nodes_per_batch,

                "gcn_layers" : interaction_layers,
                "fc_layers" : interaction_fc_layers            
            }

            self._interaction_net = GCN(**gnn_params)
            self._interaction_net_vf = GCN(**gnn_params)
            downstream_in_size =  self._dim_fc+self._interaction_obs_num*interaction_fc_out_shape
            print("Downstream Size Total: %d \nFC dim: %d, \nInteraction Output: %d"%(downstream_in_size,self._dim_fc,self._interaction_obs_num*interaction_fc_out_shape*2))
        
        elif self.interaction_net_type == "gat":
            num_nodes_per_batch = self._interaction_feature_dim[0]
            gnn_params = {
                "channel_in":self._interaction_feature_dim[1],
                "channel_out":interaction_out_channel,
                "edge_dim":self._interaction_feature_dim[1],
                "out_dim": interaction_fc_out_shape,
                "num_nodes_per_batch":num_nodes_per_batch,

                "gcn_layers" : interaction_layers,
                "fc_layers" : interaction_fc_layers            
            }

            self._interaction_net = GAT(**gnn_params)
            self._interaction_net_vf = GAT(**gnn_params)
            downstream_in_size =  self._dim_fc+self._interaction_obs_num*interaction_fc_out_shape
            print("Downstream Size Total: %d \nFC dim: %d, \nInteraction Output: %d"%(downstream_in_size,self._dim_fc,self._interaction_obs_num*interaction_fc_out_shape*2))
        
        elif self.interaction_net_type == "fc":
            downstream_in_size =  dim_state

        ''' Construct the policy function '''
        param = {
            "size_in": downstream_in_size, 
            "size_out": num_outputs, 
            "layers": policy_fn_layers,
            "append_log_std": append_log_std,
            "log_std_type": log_std_type,
            "sample_std": sample_std
        }
        self._policy_fn = FC(**param)

        ''' Construct the value function '''

        param = {
            "size_in": downstream_in_size, 
            "size_out": 1, 
            "layers": value_fn_layers,
            "append_log_std": False
        }
        self._value_fn = FC(**param)

        ''' Keep the latest output of the value function '''

        self._cur_value = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        
        obs = input_dict["obs_flat"].float()
        obs = obs.reshape(obs.shape[0], -1)
        if self.interaction_net_type=="conv":
            assert self._sparse_interaction == False ## Don't want to use the sparse state representation

            pf_outs = []
            vf_outs = []
            for i in range(self._interaction_obs_num):
                
                seg_length = self._interaction_obs_dim
                seg = obs[:,self._dim_fc+i*seg_length:self._dim_fc+(i+1)*seg_length]
                interaction_point_dim = self._interaction_feature_dim[0]*self._interaction_feature_dim[1]
                
                seg_interaction_points = seg[:,:interaction_point_dim]

                seg = seg[:, interaction_point_dim:]
                seg = seg.reshape(seg.shape[0],self._interaction_feature_dim[0],self._interaction_feature_dim[0],self._interaction_feature_dim[1])
                seg = torch.moveaxis(seg,-1,1)
                seg_pos_info = seg[:,:3,:,:]
                seg_vel_info = seg[:,3:,:,:]

                pos_out_pf = self._pos_interaction_net(seg_pos_info)
                vel_out_pf = self._vel_interaction_net(seg_vel_info)

                pos_out_vf = self._pos_interaction_net_vf(seg_pos_info)
                vel_out_vf = self._vel_interaction_net_vf(seg_vel_info)

                pf_outs.append(seg_interaction_points)
                pf_outs.append(pos_out_pf)
                pf_outs.append(vel_out_pf)

                vf_outs.append(seg_interaction_points)
                vf_outs.append(pos_out_vf)
                vf_outs.append(vel_out_vf)
                
            pf_outs.append(obs[:,:self._dim_fc])
            vf_outs.append(obs[:,:self._dim_fc])

            pf_obs = torch.cat(pf_outs, axis=-1)   
            vf_obs = torch.cat(vf_outs, axis=-1)   
        elif self.interaction_net_type == "gat":
            assert self._sparse_interaction == True ## Must use the sparse state representation
            pf_outs = []
            vf_outs = []
            max_edges = self._interaction_feature_dim[0] * self._interaction_feature_dim[0]
            
            for i in range(self._interaction_obs_num):

                seg_length = self._interaction_obs_dim
                seg = obs[:,self._dim_fc+i*seg_length:self._dim_fc+(i+1)*seg_length]

                interaction_point_dim = self._interaction_feature_dim[0]*self._interaction_feature_dim[1]
                num_edges = seg[:,interaction_point_dim].to(torch.long)

                total_num_edges = self._interaction_feature_dim[0]*self._interaction_feature_dim[0]
                seg_interaction_points = seg[:,:interaction_point_dim]
                seg_interaction_points = seg_interaction_points.reshape(seg_interaction_points.shape[0],self._interaction_feature_dim[0],self._interaction_feature_dim[1])

                seg_interaction_edges_connectivity = seg[:,interaction_point_dim+1:interaction_point_dim+1+total_num_edges*2]
                seg_interaction_edges_connectivity = seg_interaction_edges_connectivity.reshape(seg_interaction_edges_connectivity.shape[0],2,total_num_edges)
                seg_interaction_edges_connectivity = seg_interaction_edges_connectivity[:,:,:max_edges]
                seg_interaction_edges_connectivity = seg_interaction_edges_connectivity.to(torch.long)
                
                seg_interaction_edges_features = seg[:,interaction_point_dim+1+total_num_edges*2:]
                seg_interaction_edges_features = seg_interaction_edges_features.reshape(seg_interaction_edges_features.shape[0],-1,self._interaction_feature_dim[1])
                seg_interaction_edges_features = seg_interaction_edges_features[:,:max_edges,:]

                batch_edge_index = Batch.from_data_list([Data(x=seg_interaction_points[j],edge_attr=seg_interaction_edges_features[j,:num_edges[j],:],edge_index=seg_interaction_edges_connectivity[j,:,:num_edges[j]],num_nodes=self._interaction_feature_dim[0]) for j in range(seg_interaction_edges_features.shape[0])])
                
                pf_out = self._interaction_net(batch_edge_index.x,batch_edge_index.edge_index,batch_edge_index.edge_attr)
                vf_out = self._interaction_net_vf(batch_edge_index.x,batch_edge_index.edge_index,batch_edge_index.edge_attr)
                pf_outs.append(pf_out)
                vf_outs.append(vf_out)
            pf_outs.append(obs[:,:self._dim_fc])
            vf_outs.append(obs[:,:self._dim_fc])

            pf_obs = torch.cat(pf_outs, axis=-1)
            vf_obs = torch.cat(vf_outs, axis=-1)
        elif self.interaction_net_type=="gcn":
            assert self._sparse_interaction == True ## Must use the sparse state representation
            pf_outs = []
            vf_outs = []
            max_edges = self._interaction_feature_dim[0] * self._interaction_feature_dim[0]
            
            for i in range(self._interaction_obs_num):

                seg_length = self._interaction_obs_dim
                seg = obs[:,self._dim_fc+i*seg_length:self._dim_fc+(i+1)*seg_length]

                interaction_point_dim = self._interaction_feature_dim[0]*self._interaction_feature_dim[1]
                num_edges = seg[:,interaction_point_dim].to(torch.long)

                total_num_edges = self._interaction_feature_dim[0]*self._interaction_feature_dim[0]
                seg_interaction_points = seg[:,:interaction_point_dim]
                seg_interaction_points = seg_interaction_points.reshape(seg_interaction_points.shape[0],self._interaction_feature_dim[0],self._interaction_feature_dim[1])

                seg_interaction_edges_connectivity = seg[:,interaction_point_dim+1:interaction_point_dim+1+total_num_edges*2]
                seg_interaction_edges_connectivity = seg_interaction_edges_connectivity.reshape(seg_interaction_edges_connectivity.shape[0],2,total_num_edges)
                seg_interaction_edges_connectivity = seg_interaction_edges_connectivity[:,:,:max_edges]
                seg_interaction_edges_connectivity = seg_interaction_edges_connectivity.to(torch.long)
                
                seg_interaction_edges_features = seg[:,interaction_point_dim+1+total_num_edges*2:]
                seg_interaction_edges_features = seg_interaction_edges_features.reshape(seg_interaction_edges_features.shape[0],-1,self._interaction_feature_dim[1])
                seg_interaction_edges_features = seg_interaction_edges_features[:,:max_edges,:]

                batch_edge_index = Batch.from_data_list([Data(x=seg_interaction_points[j],edge_attr=seg_interaction_edges_features[j,:num_edges[j],:],edge_index=seg_interaction_edges_connectivity[j,:,:num_edges[j]],num_nodes=self._interaction_feature_dim[0]) for j in range(seg_interaction_edges_features.shape[0])])

                pf_out = self._interaction_net(batch_edge_index.x,batch_edge_index.edge_index)
                vf_out = self._interaction_net_vf(batch_edge_index.x,batch_edge_index.edge_index)
                pf_outs.append(pf_out)
                vf_outs.append(vf_out)

            pf_outs.append(obs[:,:self._dim_fc])
            vf_outs.append(obs[:,:self._dim_fc])

            pf_obs = torch.cat(pf_outs, axis=-1)
            vf_obs = torch.cat(vf_outs, axis=-1)
        else:
            assert self._sparse_interaction == False ## Don't want to use the sparse state representation
            pf_obs = obs
            vf_obs = obs
        logits = self._policy_fn(pf_obs)
        self._cur_value = self._value_fn(vf_obs).squeeze(1)

        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    def save_policy_weights(self, file):
        torch.save(self._policy_fn.state_dict(), file)
        # print(self._policy_fn.state_dict())
        # print(self._value_fn.state_dict())

    def load_policy_weights(self, file):
        self._policy_fn.load_state_dict(torch.load(file))
        self._policy_fn.eval()
    def get_attention_weight(self):
        return self._interaction_net._attention_weight
class FullyConnectedPolicy(TorchModelV2, nn.Module):
    ''' 
    A policy that generates action and value with FCNN
    '''
    DEFAULT_CONFIG = {
        "log_std_type": "constant",
        "sample_std": 1.0,

        "policy_fn_type": "mlp",
        "policy_fn_layers": [
            {"type": "fc", "hidden_size": 256, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": 256, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": "output", "activation": "linear", "init_weight": {"name": "normc", "std": 0.01}},
        ],
        
        "log_std_fn_hiddens": [64, 64],
        "log_std_fn_activations": ["relu", "relu", "linear"],
        "log_std_fn_init_weights": [1.0, 1.0, 0.01],
        "log_std_fn_base": 0.0,
        
        "value_fn_layers": [
            {"type": "fc", "hidden_size": 256, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": 256, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": "output", "activation": "linear", "init_weight": {"name": "normc", "std": 0.01}},
        ],
    }
    """Generic fully connected network."""
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **model_kwargs):

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        ''' Load and check configuarations '''

        custom_model_config = FullyConnectedPolicy.DEFAULT_CONFIG.copy()
        custom_model_config_by_user = model_config.get("custom_model_config")
        if custom_model_config_by_user:
            custom_model_config.update(custom_model_config_by_user)

        '''
        constant
            log_std will not change during the training
        state_independent
            log_std will be learned during the training
            but it does not depend on the state of the agent
        state_dependent:
            log_std will be learned during the training
            and it depens on the state of the agent
        '''

        log_std_type = custom_model_config.get("log_std_type")
        assert log_std_type in \
            ["constant", "state_independent", "state_dependent"]

        sample_std = custom_model_config.get("sample_std")
        assert np.array(sample_std).all() > 0.0, \
            "The value shoulde be positive"

        assert num_outputs % 2 == 0, (
            "num_outputs must be divisible by two", num_outputs)
        num_outputs = num_outputs//2
        append_log_std = (log_std_type != "state_dependent")

        policy_fn_type = custom_model_config.get("policy_fn_type")
        policy_fn_layers = custom_model_config.get("policy_fn_layers")
        
        value_fn_layers = custom_model_config.get("value_fn_layers")

        dim_state = int(np.product(obs_space.shape))

        ''' Construct the policy function '''

        if policy_fn_type == "mlp":
            param = {
                "size_in": dim_state, 
                "size_out": num_outputs, 
                "layers": policy_fn_layers,
                "append_log_std": append_log_std,
                "log_std_type": log_std_type,
                "sample_std": sample_std
            }
            self._policy_fn = FC(**param)
        else:
            raise NotImplementedError

        ''' Construct the value function '''

        param = {
            "size_in": dim_state, 
            "size_out": 1, 
            "layers": value_fn_layers,
            "append_log_std": False
        }
        self._value_fn = FC(**param)

        ''' Keep the latest output of the value function '''

        self._cur_value = None

        ''' Construct log_std function if necessary '''

        self._log_std_fn = None

        if log_std_type == "state_dependent":

            log_std_fn_hiddens = \
                custom_model_config.get("log_std_fn_hiddens")
            log_std_fn_activations = \
                custom_model_config.get("log_std_fn_activations")
            log_std_fn_init_weights = \
                custom_model_config.get("log_std_fn_init_weights")
            self._log_std_fn_base = np.log(sample_std)

            assert len(log_std_fn_hiddens) > 0
            assert len(log_std_fn_hiddens)+1 == len(log_std_fn_activations)
            assert len(log_std_fn_hiddens)+1 == len(log_std_fn_init_weights)

            param_log_std_fn = {
                "size_in": dim_state,
                "size_out": num_outputs,
                "hiddens": log_std_fn_hiddens,
                "activations": log_std_fn_activations,
                "init_weights": log_std_fn_init_weights,
                "append_log_std": False,
            }

            self._log_std_fn = net_cls(**param_log_std_fn)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        obs = obs.reshape(obs.shape[0], -1)
        
        logits = self._policy_fn(obs)
        self._cur_value = self._value_fn(obs).squeeze(1)

        if self._log_std_fn is not None:
            log_std = self._log_std_fn_base + self._log_std_fn(obs)
            logits = torch.cat([logits, log_std], axis=-1)
            
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    def save_policy_weights(self, file):
        torch.save(self._policy_fn.state_dict(), file)
        # print(self._policy_fn.state_dict())
        # print(self._value_fn.state_dict())

    def load_policy_weights(self, file):
        self._policy_fn.load_state_dict(torch.load(file))
        self._policy_fn.eval()
class MOEPolicyBase(TorchModelV2, nn.Module):
    ''' 
    A base policy with Mixture-of-Experts structure
    '''
    DEFAULT_CONFIG = {
        "log_std_type": "constant",
        "sample_std": 1.0,
        "expert_size_in": None,
        "expert_hiddens": [
            [128, 128],
            [128, 128],
            [128, 128],
        ],
        "expert_activations": [
            ["relu", "relu", "linear"],
            ["relu", "relu", "linear"],
            ["relu", "relu", "linear"],
        ],
        "expert_init_weights": [
            [1.0, 1.0, 0.01],
            [1.0, 1.0, 0.01],
            [1.0, 1.0, 0.01],
        ],
        "expert_log_std_types": [
            'constant',
            'constant',
            'constant',
        ],
        "expert_sample_stds": [
            0.2,
            0.2,
            0.2,
        ],
        "expert_checkpoints": [
            None,
            None,
            None,
        ],
        "expert_learnable": [
            True,
            True,
            True,
        ],
        
        "use_helper": False,
        "helper_hiddens": [
            [64, 64],
            [64, 64],
            [64, 64],
        ],
        "helper_activations": [
            ["tanh", "tanh", "tanh"],
            ["tanh", "tanh", "tanh"],
            ["tanh", "tanh", "tanh"],
        ],
        "helper_init_weights": [
            [1.0, 1.0, 0.01],
            [1.0, 1.0, 0.01],
            [1.0, 1.0, 0.01],
        ],
        "helper_checkpoints": [
            None,
            None,
            None,
        ],
        "helper_learnable": [
            True,
            True,
            True,
        ],
        "helper_range": 1.0,

        "gate_fn_type": "mlp",
        "gate_fn_hiddens": [128, 128],
        "gate_fn_activations": ["relu", "relu", "linear"],
        "gate_fn_init_weights": [1.0, 1.0, 0.01],
        "gate_fn_learnable": True,
        "gate_fn_autoreg": False,
        "gate_fn_autoreg_alpha": 0.95,
        
        "value_fn_hiddens": [128, 128],
        "value_fn_activations": ["relu", "relu", "linear"],
        "value_fn_init_weights": [1.0, 1.0, 0.01],
    }
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **model_kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        ''' Load and check configuarations '''

        assert num_outputs % 2 == 0, (
            "num_outputs must be divisible by two", num_outputs)
        dim_action_mean = num_outputs // 2
        dim_action_std = num_outputs // 2

        custom_model_config = MOEPolicyBase.DEFAULT_CONFIG.copy()
        custom_model_config_by_user = model_config.get("custom_model_config")
        if custom_model_config_by_user:
            custom_model_config.update(custom_model_config_by_user)

        self._model_config = custom_model_config

        expert_size_in        = self._model_config.get("expert_size_in")
        expert_hiddens        = self._model_config.get("expert_hiddens")
        expert_activations    = self._model_config.get("expert_activations")
        expert_init_weights   = self._model_config.get("expert_init_weights")
        expert_log_std_types  = self._model_config.get("expert_log_std_types")
        expert_sample_stds    = self._model_config.get("expert_sample_stds")
        expert_checkpoints    = self._model_config.get("expert_checkpoints")
        expert_learnable      = self._model_config.get("expert_learnable")

        use_helper            = self._model_config.get("use_helper")

        if use_helper:
            helper_hiddens        = self._model_config.get("helper_hiddens")
            helper_activations    = self._model_config.get("helper_activations")
            helper_init_weights   = self._model_config.get("helper_init_weights")
            helper_checkpoints    = self._model_config.get("helper_checkpoints")
            helper_learnable      = self._model_config.get("helper_learnable")
            self._helper_range    = self._model_config.get("helper_range")
            assert len(helper_hiddens) == len(expert_hiddens)
            assert len(helper_activations) == len(expert_activations)
            assert len(helper_init_weights) == len(expert_init_weights)
            assert len(helper_checkpoints) == len(expert_checkpoints)
            assert len(helper_learnable) == len(expert_learnable)

        gate_fn_type            = self._model_config.get("gate_fn_type")
        gate_fn_hiddens         = self._model_config.get("gate_fn_hiddens")
        gate_fn_activations     = self._model_config.get("gate_fn_activations")
        gate_fn_init_weights    = self._model_config.get("gate_fn_init_weights")
        gate_fn_autoreg         = self._model_config.get("gate_fn_autoreg")
        gate_fn_autoreg_alpha   = self._model_config.get("gate_fn_autoreg_alpha")

        value_fn_hiddens      = self._model_config.get("value_fn_hiddens")
        value_fn_activations  = self._model_config.get("value_fn_activations")
        value_fn_init_weights = self._model_config.get("value_fn_init_weights")

        project_dir           = self._model_config.get('project_dir')

        num_experts = len(expert_hiddens)
        dim_state = int(np.product(obs_space.shape))
        dim_state_expert = dim_state if expert_size_in is None else expert_size_in

        ''' Construct the gate function '''
        
        self._gate_fn_type = gate_fn_type
        self._gate_fn_autoreg = gate_fn_autoreg
        self._gate_fn_autoreg_alpha = gate_fn_autoreg_alpha

        if gate_fn_autoreg:
            assert gate_fn_type == "mlp"
            assert gate_fn_autoreg_alpha > 0.0

        if gate_fn_type == "mlp":
            self._gate_fn = FC(
                size_in=dim_state, 
                size_out=num_experts, 
                hiddens=gate_fn_hiddens, 
                activations=gate_fn_activations, 
                init_weights=gate_fn_init_weights, 
                append_log_std=False)
        elif gate_fn_type == "lstm":
            self._gate_fn_lstm_hidden_size = gate_fn_hiddens[0]
            self._gate_fn_lstm_num_layers  = len(gate_fn_hiddens)
            self._gate_fn_lstm = nn.LSTM(
                input_size=dim_state,
                hidden_size=self._gate_fn_lstm_hidden_size,
                num_layers=self._gate_fn_lstm_num_layers,
                batch_first=True,
            )
            self._gate_fn = nn.Linear(
                in_features=self._gate_fn_lstm_hidden_size,
                out_features=num_experts
            )
        else:
            raise NotImplementedError

        ''' Construct experts '''

        experts = []
        helpers = []
        for i in range(num_experts):
            append_log_std = False if expert_log_std_types[i]=='none' else True
            sample_std = expert_sample_stds[i] if append_log_std else None
            ''' Expert definition '''
            expert = FC(
                size_in=dim_state_expert, 
                size_out=dim_action_mean, 
                hiddens=expert_hiddens[i], 
                activations=expert_activations[i], 
                init_weights=expert_init_weights[i], 
                append_log_std=append_log_std,
                log_std_type=expert_log_std_types[i], 
                sample_std=sample_std,
            )
            ''' Load checkpoint if it exists '''
            if expert_checkpoints[i]:
                if project_dir:
                    checkpoint = os.path.join(
                        project_dir, expert_checkpoints[i])
                else:
                    checkpoint = expert_checkpoints[i]
                expert.load_state_dict(torch.load(checkpoint))
                expert.eval()
            ''' Set trainable '''
            for name, param in expert.named_parameters():
                param.requires_grad = expert_learnable[i]
            experts.append(expert)
            
            ''' Do the same procedure for helpers '''
            if use_helper:
                helper = FC(
                    size_in=dim_state_expert, 
                    size_out=dim_action_mean, 
                    hiddens=helper_hiddens[i], 
                    activations=helper_activations[i], 
                    init_weights=helper_init_weights[i],
                    append_log_std=False,
                )
                if helper_checkpoints[i]:
                    if project_dir:
                        checkpoint = os.path.join(
                            project_dir, helper_checkpoints[i])
                    else:
                        checkpoint = helper_checkpoints[i]
                    expert.load_state_dict(torch.load(checkpoint))
                    expert.eval()
                for name, param in helper.named_parameters():
                    param.requires_grad = helper_learnable[i]
                helpers.append(helper)
        
        self._experts = nn.ModuleList(experts)
        self._helpers = nn.ModuleList(helpers)

        ''' Construct the value function '''
        
        self._value_fn = FC(
            size_in=dim_state, 
            size_out=1, 
            hiddens=value_fn_hiddens, 
            activations=value_fn_activations, 
            init_weights=value_fn_init_weights, 
            append_log_std=False)

        self._dim_action_mean = dim_action_mean
        self._dim_action_std = dim_action_std
        self._dim_state_expert = dim_state_expert
        self._num_experts = num_experts
        self._cur_value = None
        self._cur_gate_weight = None

    @override(TorchModelV2)
    def get_initial_state(self):
        if self._gate_fn_type=="mlp":
            if self._gate_fn_autoreg:
                return torch.ones(1, self._num_experts)/self._num_experts
            else:
                return []
        elif self._gate_fn_type=="lstm":
            # # The shape should be (num_hidden_layers, hidden_size)
            h0 = self._gate_fn.weight.new(
                self._gate_fn_lstm_num_layers, 
                self._gate_fn_lstm_hidden_size).zero_()
            c0 = self._gate_fn.weight.new(
                self._gate_fn_lstm_num_layers, 
                self._gate_fn_lstm_hidden_size).zero_()
            return h0, c0
        else:
            raise NotImplementedError

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        raise NotImplementedError

    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    def gate_function(self):
        return self._cur_gate_weight

    def num_experts(self):
        return self._num_experts

    def save_weights_all(self, file):
        torch.save(self.state_dict(), file)

    def load_wegiths_all(self, file):
        self.load_state_dict(torch.load(file))
        self.eval()

    def save_weights_seperate(
        self, file_gate=None, file_experts=None, file_helpers=None):
        if file_gate:
            name = "%s.pt" % (file_gate)
            torch.save(self._gate_fn.state_dict(), name)
        if file_experts:
            for i, e in enumerate(self._experts):
                name = "%s_%02d.pt" % (file_experts, i)
                torch.save(e.state_dict(), name)
        if file_helpers:
            for i, h in enumerate(self._helpers):
                name = "%s_%02d.pt" % (file_helpers, i)
                torch.save(h.state_dict(), name)

    def load_weights_seperate(
        self, file_gate=None, file_experts=None, file_helpers=None):
        if file_gate:
            name = "%s.pt" % (file_gate)
            self._gate_fn.load_state_dict(torch.load(name))
            self._gate_fn.eval()
        if file_experts:
            for i, e in enumerate(self._experts):
                name = "%s_%02d.pt" % (file_experts, i)
                e.load_state_dict(torch.load(name))
                e.eval()
        if file_helpers:
            for i, h in enumerate(self._helpers):
                name = "%s_%02d.pt" % (file_helpers, i)
                h.load_state_dict(torch.load(name))
                h.eval()

class MOEPolicyAdditive(MOEPolicyBase):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **model_kwargs):
        super().__init__(
            obs_space, action_space, num_outputs, 
            model_config, name, **model_kwargs)

        log_std_type = self._model_config.get("log_std_type")
        assert log_std_type in ["constant", "state_independent"]

        sample_std = self._model_config.get("sample_std")
        assert sample_std > 0.0, "The value shoulde be positive"

        self._append_log_std = AppendLogStd(
            type=log_std_type,
            init_val=np.log(sample_std), 
            dim=self._dim_action_std,
        )
    
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        obs_expert = obs[...,:self._dim_state_expert]

        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        if seq_lens is not None:
            max_seq_len=torch.max(seq_lens)

        if self._gate_fn_type=="mlp":
            if self._gate_fn_autoreg:
                alpha = self._gate_fn_autoreg_alpha
                
                w_cur = softmax_normalized(self._gate_fn(obs), dim=1)
                w_cur = add_time_dimension(
                    w_cur,
                    max_seq_len=max_seq_len,
                    framework="torch",
                    time_major=False,
                )

                w = []

                for i, sl in enumerate(seq_lens):
                    w_prev_i = state[0][i]
                    for j in range(max_seq_len):
                        if j < sl:
                            w_cur_i = w_cur[i][j]
                            w_prev_i = alpha * w_prev_i + (1.0-alpha) * w_cur_i
                            w.append(w_prev_i)
                        else:
                            w.append(torch.zeros_like(state[0][i]))
                    state[0][i] = w_prev_i

                w = torch.stack(w).unsqueeze(-1)

                # alpha = self._gate_fn_autoreg_alpha
                # w_prev, w_cur = state[0], softmax_normalized(self._gate_fn(obs), dim=1)
                # w_new = (alpha * w_prev + (1.0-alpha) * w_cur)
                # w = w_new.unsqueeze(-1)
                # state = [w_new]
            else:
                w = softmax_normalized(self._gate_fn(obs), dim=1).unsqueeze(-1)
        elif self._gate_fn_type=="lstm":            
            if isinstance(seq_lens, np.ndarray):
                seq_lens = torch.Tensor(seq_lens).int()

            max_seq_len = obs.shape[0] // seq_lens.shape[0]
            input_lstm = add_time_dimension(
                obs,
                max_seq_len=max_seq_len,
                framework="torch",
                time_major=False,
            )

            ''' 
            Assume that the shape of state is 
            (batch, num_layers * num_directions, hidden_size). So we change
            the first axis with the second axis.
            '''
            h_lstm, c_lstm = state[0], state[1]
            h_lstm = h_lstm.reshape(h_lstm.shape[1], h_lstm.shape[0], h_lstm.shape[2])
            c_lstm = c_lstm.reshape(c_lstm.shape[1], c_lstm.shape[0], c_lstm.shape[2])
            
            output_lstm, (h_lstm, c_lstm) = self._gate_fn_lstm(input_lstm, (h_lstm, c_lstm))
            output_lstm = output_lstm.reshape(-1, output_lstm.shape[-1])
            
            '''
            Change the first and second axes of the output state so that
            it matches to the assumption
            '''
            h_lstm = h_lstm.reshape(h_lstm.shape[1], h_lstm.shape[0], h_lstm.shape[2])
            c_lstm = c_lstm.reshape(c_lstm.shape[1], c_lstm.shape[0], c_lstm.shape[2])
            
            state = [h_lstm, c_lstm]

            w = softmax_normalized(self._gate_fn(output_lstm), dim=1).unsqueeze(-1)
        else:
            raise NotImplementedError
        
        x = torch.stack([e(obs_expert) for e in self._experts], dim=1)
        if len(self._helpers) > 0:
            x += torch.stack(
                [self._helper_range*h(obs_expert) for h in self._helpers], 
                dim=1)
        logits = self._append_log_std(torch.sum(w*x, dim=1))

        self._cur_gate_weight = w
        self._cur_value = self._value_fn(obs).squeeze(1)
        
        return logits, state

class MOEPolicyMultiplicative(MOEPolicyBase):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **model_kwargs):
        super().__init__(
            obs_space, action_space, num_outputs, 
            model_config, name, **model_kwargs)
    
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        obs_expert = obs[...,:self._dim_state_expert]
        
        w = softmax_normalized(self._gate_fn(obs), dim=1).unsqueeze(-1)
        x = torch.stack([expert(obs_expert) for expert in self._experts], dim=1)

        expert_mean = x[...,:self.num_outputs]
        expert_std = torch.exp(x[...,self.num_outputs:])

        z = w / expert_std
        std = 1.0 / torch.sum(z, dim=1)
        logstd = torch.log(std)
        mean = std * torch.sum(z * expert_mean, dim=1)
        
        logits = torch.concat([], )

        self._cur_gate_weight = w
        self._cur_value = self._value_fn(obs).squeeze(1)
        
        return logits, state

class LSTMPolicy(TorchModelV2, nn.Module):
    ''' 
    A policy that generates action and value with RNN (LSTM)
    '''
    DEFAULT_CONFIG = {
        "log_std_type": "constant",
        "sample_std": 1.0,

        "lstm_hidden_size": 128,
        "lstm_num_layers": 2,

        "decoder_hiddens": [128],
        "decoder_activations": ["relu", "linear"],
        "decoder_init_weights": [1.0, 0.01],
        
        "value_fn_hiddens": [128, 128],
        "value_fn_activations": ["relu", "relu", "linear"],
        "value_fn_init_weights": [1.0, 1.0, 0.01],
    }
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **model_kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        assert num_outputs % 2 == 0, (
            "num_outputs must be divisible by two", num_outputs)
        num_outputs = num_outputs // 2

        ''' Load and check configuarations '''

        custom_model_config = LSTMPolicy.DEFAULT_CONFIG.copy()
        custom_model_config_by_user = model_config.get("custom_model_config")
        if custom_model_config_by_user:
            custom_model_config.update(custom_model_config_by_user)

        log_std_type = \
            custom_model_config.get("log_std_type")
        sample_std = \
            custom_model_config.get("sample_std")

        assert log_std_type in ["constant", "state_independent"]
        assert sample_std > 0.0

        lstm_hidden_size = \
            custom_model_config.get("lstm_hidden_size")
        lstm_num_layers  = \
            custom_model_config.get("lstm_num_layers")

        assert lstm_hidden_size > 0
        assert lstm_num_layers > 0

        decoder_hiddens = \
            custom_model_config.get("decoder_hiddens")
        decoder_activations = \
            custom_model_config.get("decoder_activations")
        decoder_init_weights = \
            custom_model_config.get("decoder_init_weights")

        assert len(decoder_hiddens) > 0
        assert len(decoder_hiddens)+1 == len(decoder_activations)
        assert len(decoder_hiddens)+1 == len(decoder_init_weights)

        value_fn_hiddens = \
            custom_model_config.get("value_fn_hiddens")
        value_fn_activations = \
            custom_model_config.get("value_fn_activations")
        value_fn_init_weights = \
            custom_model_config.get("value_fn_init_weights")

        assert len(value_fn_hiddens) > 0
        assert len(value_fn_hiddens)+1 == len(value_fn_activations)
        assert len(value_fn_hiddens)+1 == len(value_fn_init_weights)

        dim_state = int(np.product(obs_space.shape))

        ''' Prepare a LSTM '''

        input_size = dim_state
        self._lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True)
        
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_input_size = input_size

        ''' Prepare a decoder '''

        param = {
            "size_in": lstm_hidden_size, 
            "size_out": num_outputs, 
            "hiddens": decoder_hiddens, 
            "activations": decoder_activations, 
            "init_weights": decoder_init_weights, 
            "append_log_std": True,
            "log_std_type": log_std_type,
            "sample_std": sample_std
        }
        self._decoder = FC(**param)

        ''' Prepare a value function '''

        param = {
            "size_in": dim_state, 
            "size_out": 1, 
            "hiddens": value_fn_hiddens, 
            "activations": value_fn_activations, 
            "init_weights": value_fn_init_weights, 
            "append_log_std": False
        }

        self._value_branch = FC(**param)

        self._cur_value = None

    @override(TorchModelV2)
    def get_initial_state(self):
        
        # The shape should be (num_hidden_layers, hidden_size)
        h0 = self._decoder._model[0]._model[0].weight.new(
            self.lstm_num_layers, self.lstm_hidden_size).zero_()
        c0 = self._decoder._model[0]._model[0].weight.new(
            self.lstm_num_layers, self.lstm_hidden_size).zero_()
        return [h0, c0]

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()

        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        if seq_lens is not None:
            max_seq_len=torch.max(seq_lens)

        input_lstm = add_time_dimension(
            obs,
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=False,
        )

        ''' 
        Assume that the shape of state is 
        (batch, num_layers * num_directions, hidden_size). So we change
        the first axis with the second axis.
        '''
        h_lstm, c_lstm = state[0], state[1]
        h_lstm = h_lstm.reshape(h_lstm.shape[1], h_lstm.shape[0], h_lstm.shape[2])
        c_lstm = c_lstm.reshape(c_lstm.shape[1], c_lstm.shape[0], c_lstm.shape[2])
        
        output_lstm, (h_lstm, c_lstm) = \
            self._lstm(input_lstm, (h_lstm, c_lstm))
        output_lstm = output_lstm.reshape(-1, output_lstm.shape[-1])
        '''
        Change the first and second axes of the output state so that
        it matches to the assumption
        '''
        h_lstm = h_lstm.reshape(h_lstm.shape[1], h_lstm.shape[0], h_lstm.shape[2])
        c_lstm = c_lstm.reshape(c_lstm.shape[1], c_lstm.shape[0], c_lstm.shape[2])
        
        state = [h_lstm, c_lstm]
        
        logits = self._decoder(output_lstm)
        self._cur_value = self._value_branch(obs).squeeze(1)
        
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value


class TaskAgnosticPolicyType1(TorchModelV2, nn.Module):
# class TaskAgnosticPolicyType1(RecurrentTorchModel, nn.Module):
    DEFAULT_CONFIG = {
        "project_dir": None,
        
        "log_std_type": "constant",
        "sample_std": 1.0,

        "load_weights": None,

        "task_encoder_type": "mlp",
        "task_encoder_inputs": ["body", "task"],
        "task_encoder_layers": [
            {"type": "fc", "hidden_size": 128, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": 128, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": "output", "activation": "linear", "init_weight": {"name": "normc", "std": 0.01}},
        ],
        "task_encoder_load_weights": None,
        "task_encoder_learnable": True,
        "task_encoder_output_dim": 32,
        "task_encoder_autoreg": False,
        "task_encoder_autoreg_alpha": 0.95,
        "task_encoder_vae": False,

        "task_decoder_enable": False,
        "task_decoder_type": "mlp",
        "task_decoder_layers": [
            {"type": "fc", "hidden_size": 128, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": 128, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": "output", "activation": "linear", "init_weight": {"name": "normc", "std": 0.01}},
        ],
        "task_decoder_load_weights": None,
        "task_decoder_learnable": True,
        
        "body_encoder_enable": False,
        "body_encoder_layers": [
            {"type": "fc", "hidden_size": 128, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": 128, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": "output", "activation": "linear", "init_weight": {"name": "normc", "std": 0.01}},
        ],
        "body_encoder_load_weights": None,
        "body_encoder_output_dim": 32,
        "body_encoder_learnable": True,

        "motor_decoder_type": "mlp",
        "motor_decoder_inputs": ["body", "task"],
        "motor_decoder_layers": [
            {"type": "fc", "hidden_size": 128, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": 128, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": "output", "activation": "linear", "init_weight": {"name": "normc", "std": 0.01}},
        ],
        "motor_decoder_load_weights": None,
        "motor_decoder_learnable": True,
        "motor_decoder_task_bypass": False,
        "motor_decoder_gate_fn_inputs": ["body", "task"],
        "motor_decoder_gate_fn_layers": [
            {"type": "fc", "hidden_size": 128, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": 128, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": 4, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": "output", "activation": "linear", "init_weight": {"name": "normc", "std": 0.01}},
            {"type": "softmax"}
        ],

        "motor_decoder_helper_enable": False,
        "motor_decoder_helper_type": "mlp",
        "motor_decoder_helper_layers": [
            {"type": "fc", "hidden_size": 128, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": 128, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": "output", "activation": "tanh", "init_weight": {"name": "normc", "std": 0.01}},
        ],
        "motor_decoder_helper_load_weights": None,
        "motor_decoder_helper_learnable": True,
        "motor_decoder_helper_range": 0.5,
        
        "value_fn_type": "mlp",
        "value_fn_layers": [
            {"type": "fc", "hidden_size": 256, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": 256, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": "output", "activation": "linear", "init_weight": {"name": "normc", "std": 0.01}},
        ],

        "future_pred_enable": False,
        "future_pred_type": "mlp",
        "future_pred_layers": [
            {"type": "fc", "hidden_size": 128, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": 128, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": "output", "activation": "linear", "init_weight": {"name": "normc", "std": 0.01}},
        ],
        "future_pred_load_weights": None,
        "future_pred_learnable": True,

        "past_pred_enable": False,
        "past_pred_type": "mlp",
        "past_pred_layers": [
            {"type": "fc", "hidden_size": 128, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": 128, "activation": "relu", "init_weight": {"name": "normc", "std": 1.0}},
            {"type": "fc", "hidden_size": "output", "activation": "linear", "init_weight": {"name": "normc", "std": 0.01}},
        ],
        "past_pred_load_weights": None,
        "past_pred_learnable": True,

        "observation_space_body": None,
        "observation_space_task": None,
    }
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **model_kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        ''' Load and check configuarations '''

        assert num_outputs % 2 == 0, (
            "num_outputs must be divisible by two", num_outputs)
        num_outputs = num_outputs // 2

        custom_model_config = TaskAgnosticPolicyType1.DEFAULT_CONFIG.copy()
        custom_model_config_by_user = model_config.get("custom_model_config")
        if custom_model_config_by_user:
            custom_model_config.update(custom_model_config_by_user)

        log_std_type = custom_model_config.get("log_std_type")
        assert log_std_type in ["constant", "state_independent"]
        sample_std = custom_model_config.get("sample_std")
        
        load_weights = custom_model_config.get("load_weights")

        project_dir = custom_model_config.get("project_dir")

        task_encoder_type = custom_model_config.get("task_encoder_type")
        task_encoder_inputs = custom_model_config.get("task_encoder_inputs")
        task_encoder_output_dim = custom_model_config.get("task_encoder_output_dim")
        task_encoder_load_weights = custom_model_config.get("task_encoder_load_weights")
        task_encoder_learnable = custom_model_config.get("task_encoder_learnable")
        task_encoder_autoreg = custom_model_config.get("task_encoder_autoreg")
        task_encoder_autoreg_alpha = custom_model_config.get("task_encoder_autoreg_alpha")
        task_encoder_vae = custom_model_config.get("task_encoder_vae")
        task_encoder_layers = custom_model_config.get("task_encoder_layers")

        task_decoder_enable = custom_model_config.get("task_decoder_enable")
        task_decoder_type = custom_model_config.get("task_decoder_type")
        task_decoder_layers = custom_model_config.get("task_decoder_layers")
        task_decoder_load_weights = custom_model_config.get("task_decoder_load_weights")
        task_decoder_learnable = custom_model_config.get("task_decoder_learnable")

        self._task_encoder_type = task_encoder_type
        self._task_encoder_inputs = task_encoder_inputs
        self._task_encoder_layers = task_encoder_layers
        self._task_encoder_output_dim = task_encoder_output_dim
        self._task_encoder_autoreg = task_encoder_autoreg
        self._task_encoder_autoreg_alpha = task_encoder_autoreg_alpha
        # self._task_encoder_hiddens = task_encoder_hiddens
        self._task_encoder_vae = task_encoder_vae

        self._task_decoder_type = task_decoder_type

        body_encoder_enable = custom_model_config.get("body_encoder_enable")
        body_encoder_type = custom_model_config.get("body_encoder_type")
        body_encoder_output_dim = custom_model_config.get("body_encoder_output_dim")
        body_encoder_load_weights = custom_model_config.get("body_encoder_load_weights")
        body_encoder_learnable = custom_model_config.get("body_encoder_learnable")
        body_encoder_layers = custom_model_config.get("body_encoder_layers")

        motor_decoder_type = custom_model_config.get("motor_decoder_type")
        motor_decoder_inputs = custom_model_config.get("motor_decoder_inputs")
        motor_decoder_layers = custom_model_config.get("motor_decoder_layers")
        motor_decoder_load_weights = custom_model_config.get("motor_decoder_load_weights")
        motor_decoder_learnable = custom_model_config.get("motor_decoder_learnable")
        motor_decoder_task_bypass = custom_model_config.get("motor_decoder_task_bypass")
        motor_decoder_gate_fn_inputs = custom_model_config.get("motor_decoder_gate_fn_inputs")
        motor_decoder_gate_fn_layers = custom_model_config.get("motor_decoder_gate_fn_layers")

        motor_decoder_helper_enable = custom_model_config.get("motor_decoder_helper_enable")
        motor_decoder_helper_type = custom_model_config.get("motor_decoder_helper_type")
        motor_decoder_helper_layers = custom_model_config.get("motor_decoder_helper_layers")
        motor_decoder_helper_load_weights = custom_model_config.get("motor_decoder_helper_load_weights")
        motor_decoder_helper_learnable = custom_model_config.get("motor_decoder_helper_learnable")
        motor_decoder_helper_range = custom_model_config.get("motor_decoder_helper_range")

        self._motor_decoder_type = motor_decoder_type
        self._motor_decoder_inputs = motor_decoder_inputs
        self._motor_decoder_task_bypass = motor_decoder_task_bypass
        self._motor_decoder_gate_fn_inputs = motor_decoder_gate_fn_inputs

        self._motor_decoder_helper_type = motor_decoder_helper_type
        self._motor_decoder_helper_range = motor_decoder_helper_range

        value_fn_type = custom_model_config.get("value_fn_type")
        value_fn_layers = custom_model_config.get("value_fn_layers")

        self._value_fn_type = value_fn_type
        self._value_fn_layers = value_fn_layers

        future_pred_enable = custom_model_config.get("future_pred_enable")
        future_pred_type = custom_model_config.get("future_pred_type")
        future_pred_layers = custom_model_config.get("future_pred_layers")
        future_pred_load_weights = custom_model_config.get("future_pred_load_weights")
        future_pred_learnable = custom_model_config.get("future_pred_learnable")

        past_pred_enable = custom_model_config.get("past_pred_enable")
        past_pred_type = custom_model_config.get("past_pred_type")
        past_pred_layers = custom_model_config.get("past_pred_layers")
        past_pred_load_weights = custom_model_config.get("past_pred_load_weights")
        past_pred_learnable = custom_model_config.get("past_pred_learnable")

        if project_dir:
            if load_weights:
                load_weights = \
                    os.path.join(project_dir, load_weights)
                assert load_weights
            if task_encoder_load_weights:
                task_encoder_load_weights = \
                    os.path.join(project_dir, task_encoder_load_weights)
                assert task_encoder_load_weights
            if task_decoder_load_weights:
                task_decoder_load_weights = \
                    os.path.join(project_dir, task_decoder_load_weights)
                assert task_decoder_load_weights
            if body_encoder_load_weights:
                body_encoder_load_weights = \
                    os.path.join(project_dir, body_encoder_load_weights)
                assert body_encoder_load_weights
            if motor_decoder_load_weights:
                motor_decoder_load_weights = \
                    os.path.join(project_dir, motor_decoder_load_weights)
                assert motor_decoder_load_weights
            if motor_decoder_helper_load_weights:
                motor_decoder_helper_load_weights = \
                    os.path.join(project_dir, motor_decoder_helper_load_weights)
                assert motor_decoder_helper_load_weights
            if future_pred_load_weights:
                future_pred_load_weights = \
                    os.path.join(project_dir, future_pred_load_weights)
                assert future_pred_load_weights
            if past_pred_load_weights:
                past_pred_load_weights = \
                    os.path.join(project_dir, past_pred_load_weights)
                assert past_pred_load_weights

        self.dim_state_body = \
            int(np.product(custom_model_config.get("observation_space_body").shape))
        self.dim_state_task = \
            int(np.product(custom_model_config.get("observation_space_task").shape))
        self.dim_state = int(np.product(obs_space.shape))
        self.dim_action = int(np.product(action_space.shape))

        assert self.dim_state == self.dim_state_body + self.dim_state_task

        size_in_task_encoder = 0
        if "body" in self._task_encoder_inputs:
            size_in_task_encoder += self.dim_state_body
        if "task" in self._task_encoder_inputs:
            size_in_task_encoder += self.dim_state_task

        size_out_task_encoder = \
            2*task_encoder_output_dim if task_encoder_vae else task_encoder_output_dim

        ''' Prepare task encoder that outputs task embedding z given s_task '''
        self._task_encoder, self._task_encoder_lstm = \
            create_layer(
                layer_type=task_encoder_type,
                layers=task_encoder_layers,
                size_in=size_in_task_encoder,
                size_out=size_out_task_encoder,
                )

        self._task_decoder = None
        self._task_decoder_lstm = None
        if task_decoder_enable:
            self._task_decoder, self._task_decoder_lstm = \
                create_layer(
                    layer_type=task_encoder_type,
                    layers=task_decoder_layers,
                    size_in=task_encoder_output_dim,
                    size_out=size_in_task_encoder,
                    )

        ''' Prepare body encoder that outputs body embedding z given s_task '''
        self._body_encoder = None
        self._body_encoder_lstm = None
        if body_encoder_enable:
            self._body_encoder, self._body_encoder_lstm = \
                create_layer(
                    layer_type=body_encoder_type,
                    layers=body_encoder_layers,
                    size_in=self.dim_state_body,
                    size_out=body_encoder_output_dim,
                    )   

        size_in_motor_decoder_body = 0
        size_in_motor_decoder_task = 0
        if "body" in self._motor_decoder_inputs:
            size_in_motor_decoder_body = body_encoder_output_dim if self._body_encoder else self.dim_state_body
        if "task" in self._motor_decoder_inputs:
            size_in_motor_decoder_task = task_encoder_output_dim if self._task_encoder else self.dim_state_task
        size_in_motor_decoder = size_in_motor_decoder_body + size_in_motor_decoder_task
        assert size_in_motor_decoder > 0

        ''' Prepare motor control decoder that outputs a given (z, s_proprioception) '''

        if self._motor_decoder_task_bypass:
            assert "body" in self._motor_decoder_inputs
            assert "task" in self._motor_decoder_inputs

        def motor_decoder_fn():
            param = {
                "size_out": num_outputs, 
                "layers": motor_decoder_layers,
                "append_log_std": True,
                "log_std_type": log_std_type, 
                "sample_std": sample_std,
            }
            if self._motor_decoder_task_bypass:
                param["size_in_x"] = size_in_motor_decoder_body
                param["size_in_z"] = size_in_motor_decoder_task
                return FC_Bypass(**param)
            else:
                param["size_in"] = size_in_motor_decoder
                return FC(**param)

        if motor_decoder_type == "mlp":
            self._motor_decoder = motor_decoder_fn()
        elif motor_decoder_type == "moe":
            size_in_gate_fn_body = 0
            size_in_gate_fn_task = 0
            if "body" in self._motor_decoder_gate_fn_inputs:
                size_in_gate_fn_body = body_encoder_output_dim if self._body_encoder else self.dim_state_body
            if "task" in self._motor_decoder_gate_fn_inputs:
                size_in_gate_fn_task = task_encoder_output_dim if self._task_encoder else self.dim_state_task
            size_in_gate_fn = size_in_gate_fn_body + size_in_gate_fn_task
            assert size_in_gate_fn > 0
            
            num_experts = motor_decoder_gate_fn_layers[-2]["hidden_size"]
            param = {
                "size_in": size_in_gate_fn, 
                "size_out": num_experts, 
                "layers": motor_decoder_gate_fn_layers,
                "append_log_std": False,
            }
            gate_fn = FC(**param)
            experts = [motor_decoder_fn() for i in range(num_experts)]
            self._motor_decoder = nn.ModuleList(experts+[gate_fn])
            self._motor_decoder_experts = experts
            self._motor_decoder_gate_fn = gate_fn
        else:
            raise NotImplementedError

        self._motor_decoder_helper = None
        if motor_decoder_helper_enable:
            assert motor_decoder_helper_layers[-1]["activation"] == "tanh"
            assert motor_decoder_helper_range > 0
            if motor_decoder_helper_type == "mlp":
                param = {
                    "size_in": size_in_motor_decoder,
                    "size_out": num_outputs, 
                    "layers": motor_decoder_helper_layers,
                    "append_log_std": False,
                }
                self._motor_decoder_helper = FC(**param)
            else:
                raise NotImplementedError

        self._future_pred = None
        if future_pred_enable:
            size_in = self.dim_action + self.dim_state_body
            size_out = self.dim_state_body
            if future_pred_type == "mlp":
                param = {
                    "size_in": size_in,
                    "size_out": size_out, 
                    "layers": future_pred_layers,
                    "append_log_std": False,
                }
                self._future_pred = FC(**param)
            else:
                raise NotImplementedError

        self._past_pred = None
        if past_pred_enable:
            size_in = self.dim_action + self.dim_state_body
            size_out = self.dim_state_body
            if past_pred_type == "mlp":
                param = {
                    "size_in": size_in,
                    "size_out": size_out, 
                    "layers": past_pred_layers,
                    "append_log_std": False,
                }
                self._past_pred = FC(**param)
            else:
                raise NotImplementedError

        ''' Prepare a value function '''

        self._value_branch, self._value_branch_lstm = \
            create_layer(
                layer_type=value_fn_type,
                layers=value_fn_layers,
                size_in=self.dim_state,
                size_out=1,
                )

        self._cur_value = None

        self._cur_body_encoder_variable = None
        self._cur_task_encoder_variable = None
        self._cur_motor_decoder_expert_weights = None
        self._cur_task_encoder_mu = None
        self._cur_task_encoder_logvar = None

        self.vae_noise = True

        ''' Load pre-trained weight if exists '''

        if load_weights:
            self.load_weights(load_weights)
            print("load_weights:", load_weights)

        if task_encoder_load_weights:
            self.load_weights_task_encoder(task_encoder_load_weights)
            self.set_learnable_task_encoder(task_encoder_learnable)
            # print("task_encoder_load_weights:", task_encoder_load_weights)
            # print("task_encoder_learnable:", task_encoder_learnable)

        if task_decoder_load_weights:
            self.load_weights_task_decoder(task_decoder_load_weights)
            self.set_learnable_task_decoder(task_decoder_learnable)
            # print("task_decoder_load_weights:", task_decoder_load_weights)
            # print("task_decoder_learnable:", task_decoder_learnable)

        if body_encoder_load_weights:
            self.load_weights_body_encoder(body_encoder_load_weights)
            self.set_learnable_body_encoder(body_encoder_learnable)
            # print("body_encoder_load_weights:", body_encoder_load_weights)
            # print("body_encoder_learnable:", body_encoder_learnable)

        if motor_decoder_load_weights:
            self.load_weights_motor_decoder(motor_decoder_load_weights)
            self.set_learnable_motor_decoder(motor_decoder_learnable)
            # print("motor_decoder_load_weights:", motor_decoder_load_weights)
            # print("motor_decoder_learnable:", motor_decoder_learnable)

        if motor_decoder_helper_load_weights:
            self.load_weights_motor_decoder_helper(motor_decoder_helper_load_weights)
            self.set_learnable_motor_decoder_helper(motor_decoder_helper_learnable)
            # print("motor_decoder_helper_load_weights:", motor_decoder_helper_load_weights)
            # print("motor_decoder_helper_learnable:", motor_decoder_helper_learnable)

        if future_pred_load_weights:
            self.load_weights_future_pred(future_pred_load_weights)
            self.set_learnable_future_pred(future_pred_learnable)
            print("future_pred_load_weights:", future_pred_load_weights)
            print("future_pred_learnable:", future_pred_learnable)

        if past_pred_load_weights:
            self.load_weights_past_pred(past_pred_load_weights)
            self.set_learnable_past_pred(past_pred_learnable)
            # print("past_pred_load_weights:", past_pred_load_weights)
            # print("past_pred_learnable:", past_pred_learnable)

    @override(TorchModelV2)
    def get_initial_state(self):
        state = []

        def get_initial_state_for_lstm(output_layer, num_layers, hidden_size):
            if isinstance(output_layer, nn.Sequential):
                layer = output_layer[0]
            elif isinstance(output_layer, FC):
                layer = output_layer._model[0]._model[0]
            else:
                raise NotImplementedError
            h0 = layer.weight.new(num_layers, hidden_size).zero_()
            c0 = layer.weight.new(num_layers, hidden_size).zero_()
            return h0, c0

        if self._task_encoder_type == "lstm":
            h0, c0 = get_initial_state_for_lstm(
                self._task_encoder, 
                self._task_encoder_layers[0]["num_layers"],
                self._task_encoder_layers[0]["hidden_size"]
                )
            state.append(h0)
            state.append(c0)
        
        if self._task_encoder_autoreg:
            state.append(torch.zeros(1, self._task_encoder_output_dim))

        if self._value_fn_type == "lstm":
            h0, c0 = get_initial_state_for_lstm(
                self._value_branch, 
                self._value_fn_layers[0]["num_layers"],
                self._value_fn_layers[0]["hidden_size"]
                )
            state.append(h0)
            state.append(c0)

        return state

    def _reparameterize(self, mu, logvar):
        if self.vae_noise:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        state_cnt = 0
        obs = input_dict["obs_flat"].float()
        z_body, z_task, state_cnt = self.forward_encoder(
            obs,
            state,
            seq_lens,
            state_cnt)
        z_body, z_task, state_cnt = self.forward_latent(
            z_body,
            z_task,
            state,
            seq_lens,
            state_cnt,
        )
        logits, state_cnt = self.forward_decoder(
            z_body,
            z_task,
            state,
            seq_lens,
            state_cnt,
        )
        future_pred, past_pred = self.forward_prediction(obs, logits)

        val, state_cnt = self.forward_value_branch(
            obs,
            state,
            seq_lens,
            state_cnt)
        
        self._cur_body_encoder_variable = z_body
        self._cur_task_encoder_variable = z_task
        self._cur_value = val.squeeze(1)
        self._cur_future_pred = future_pred
        self._cur_past_pred = past_pred

        return logits, state

    def forward_encoder(self, obs, state, seq_lens, state_cnt):
        ''' Assume state==(state_body, state_task) '''
        
        if "body" in self._task_encoder_inputs and "task" in self._task_encoder_inputs:
            obs_task = obs[...,:]
        elif "body" in self._task_encoder_inputs:
            obs_task = obs[...,:self.dim_state_body]
        elif "task" in self._task_encoder_inputs:
            obs_task = obs[...,self.dim_state_body:]
        else:
            raise NotImplementedError
        
        obs_body = obs[...,:self.dim_state_body]
        z_body = self._body_encoder(obs_body) if self._body_encoder else obs_body

        z_task, state_cnt = forward_layer(
            obs_task,
            seq_lens,
            state, 
            state_cnt,
            self._task_encoder,
            self._task_encoder_lstm)

        if self._task_encoder_vae:
            mu = z_task[...,:self._task_encoder_output_dim]
            logvar = z_task[...,self._task_encoder_output_dim:]
            z_task = self._reparameterize(mu, logvar)
            self._cur_task_encoder_mu = mu
            self._cur_task_encoder_logvar = logvar

        return z_body, z_task, state_cnt

    def forward_latent(self, z_body, z_task, state, seq_lens, state_cnt):
        if self._task_encoder_autoreg:
            if isinstance(seq_lens, np.ndarray):
                seq_lens = torch.Tensor(seq_lens).int()
            if seq_lens is not None:
                max_seq_len=torch.max(seq_lens)

            state_autoreg = state[state_cnt]
            alpha = self._task_encoder_autoreg_alpha
            
            z_task_cur = add_time_dimension(
                z_task,
                max_seq_len=max_seq_len,
                framework="torch",
                time_major=False,
            )

            z_task_list = []

            for i, sl in enumerate(seq_lens):
                z_task_prev_i = state_autoreg[i][0]
                for j in range(max_seq_len):
                    if j < sl:
                        z_task_cur_i = z_task_cur[i][j]
                        z_task_prev_i = alpha * z_task_prev_i + (1.0-alpha) * z_task_cur_i
                        z_task_list.append(z_task_prev_i)
                    else:
                        z_task_list.append(torch.zeros_like(z_task_prev_i))
                state_autoreg[i][0] = z_task_prev_i

            z_task = torch.stack(z_task_list)

            state_cnt += 1
        
        return z_body, z_task, state_cnt


    def forward_decoder(self, z_body, z_task, state, seq_lens, state_cnt):
        if self._task_decoder:
            if self._task_decoder_type == "mlp":
                self._cur_task_decoder_out = self._task_decoder(z_task)
            else:
                raise NotImplementedError
        
        if self._motor_decoder:

            z = []
            if "body" in self._motor_decoder_inputs:
                z.append(z_body)
            if "task" in self._motor_decoder_inputs:
                z.append(z_task)
            assert len(z) > 0
            z = torch.cat(z, axis=-1)

            def eval(fn, z, z_body, z_task):
                if self._motor_decoder_task_bypass:
                    return fn(z_body, z_task)
                else:
                    return fn(z)

            if self._motor_decoder_type == "mlp":
                logits = eval(self._motor_decoder, z, z_body, z_task)
            elif self._motor_decoder_type == "moe":
                z_gate_fn = []
                if "body" in self._motor_decoder_gate_fn_inputs:
                    z_gate_fn.append(z_body)
                if "task" in self._motor_decoder_gate_fn_inputs:
                    z_gate_fn.append(z_task)
                assert len(z_gate_fn) > 0
                z_gate_fn = torch.cat(z_gate_fn, axis=-1)
                
                w = self._motor_decoder_gate_fn(z_gate_fn)
                x = torch.stack([eval(expert, z, z_body, z_task) for expert in self._motor_decoder_experts], dim=1)
                
                # w = self._motor_decoder_gate_fn(z_gate_fn).unsqueeze(-1)
                # x = torch.stack([expert(z) for expert in self._motor_decoder_experts], dim=1)
                
                logits = torch.sum(w.unsqueeze(-1)*x, dim=1)
                self._cur_motor_decoder_expert_weights = w
                # print("***************************")
                # print(w)
                # # print(z_gate_fn)
                # print("***************************")
            else:
                raise NotImplementedError

            if self._motor_decoder_helper:
                if self._motor_decoder_helper_type == "mlp":
                    logits_add = eval(self._motor_decoder_helper, z, z_body, z_task)
                else:
                    raise NotImplementedError
                logits[..., :self.dim_action] += self._motor_decoder_helper_range*logits_add
        
        return logits, state_cnt

    def forward_prediction(self, obs, logits):
        future_pred, past_pred = None, None
        if self._future_pred:
            x = torch.cat([obs[...,:self.dim_state_body], logits[...,:self.dim_action]], axis=-1)
            future_pred = self._future_pred(x)
            # print(future_pred)
        if self._past_pred:
            x = torch.cat([obs[...,self.dim_state_body:], logits[...,:self.dim_action]], axis=-1)
            past_pred = self._past_pred(x)
        return future_pred, past_pred

    def forward_value_branch(self, obs, state, seq_lens, state_cnt):
        val, state_cnt = forward_layer(
            obs,
            seq_lens,
            state, 
            state_cnt,
            self._value_branch,
            self._value_branch_lstm)
        return val, state_cnt

    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    def set_exploration_std(self, std):
        log_std = np.log(std)
        if self._motor_decoder_type == "mlp":
            self._motor_decoder._model[-1].set_val(log_std)
        elif self._motor_decoder_type == "moe":
            for expert in self._motor_decoder:
                expert._model[-1].set_val(log_std)
        else:
            raise NotImplementedError

    def task_encoder_variable(self):
        return self._cur_task_encoder_variable

    def body_encoder_variable(self):
        return self._cur_body_encoder_variable

    def save_weights(self, file):
        torch.save(self.state_dict(), file)

    def load_weights(self, file):
        self.load_state_dict(torch.load(file))
        self.eval()

    def save_weights_task_encoder(self, file):
        state_dict = {}
        if self._task_encoder:
            state_dict['task_encoder'] = \
                self._task_encoder.state_dict()
        if self._task_encoder_lstm:
            state_dict['task_encoder_lstm'] = \
                self._task_encoder_lstm.state_dict()
        torch.save(state_dict, file)

    def load_weights_task_encoder(self, file):
        state_dict = torch.load(file)
        if self._task_encoder:
            self._task_encoder.load_state_dict(state_dict['task_encoder'])
            self._task_encoder.eval()
        if self._task_encoder_lstm:
            self._task_encoder_lstm.load_state_dict(state_dict['task_encoder_lstm'])
            self._task_encoder_lstm.eval()

    def save_weights_task_decoder(self, file):
        if self._task_decoder:
            torch.save(self._task_decoder.state_dict(), file)

    def load_weights_task_decoder(self, file):
        if self._task_decoder:
            self._task_decoder.load_state_dict(torch.load(file))
            self._task_decoder.eval()

    def save_weights_body_encoder(self, file):
        if self._body_encoder:
            torch.save(self._body_encoder.state_dict(), file)

    def load_weights_body_encoder(self, file):
        if self._body_encoder:
            self._body_encoder.load_state_dict(torch.load(file))
            self._body_encoder.eval()

    def save_weights_motor_decoder(self, file):
        if self._motor_decoder:
            torch.save(self._motor_decoder.state_dict(), file)

    def save_weights_motor_decoder_helper(self, file):
        if self._motor_decoder_helper:
            torch.save(self._motor_decoder_helper.state_dict(), file)

    def load_weights_motor_decoder(self, file):
        if self._motor_decoder:
            ''' Ignore weights of log_std for valid exploration '''
            dict_weights_orig = self._motor_decoder.state_dict()
            dict_weights_loaded = torch.load(file)
            for key in dict_weights_loaded.keys():
                if 'log_std' in key:
                    dict_weights_loaded[key] = dict_weights_orig[key]
                    # print(dict_weights_orig[key])
            self._motor_decoder.load_state_dict(dict_weights_loaded)
            self._motor_decoder.eval()

    def load_weights_motor_decoder_helper(self, file):
        if self._motor_decoder_helper:
            self._motor_decoder_helper.load_state_dict(torch.load(file))
            self._motor_decoder_helper.eval()

    def save_weights_future_pred(self, file):
        if self._future_pred:
            torch.save(self._future_pred.state_dict(), file)

    def load_weights_future_pred(self, file):
        if self._future_pred:
            self._future_pred.load_state_dict(torch.load(file))
            self._future_pred.eval()

    def set_learnable_task_encoder(self, learnable):
        if self._task_encoder:
            for name, param in self._task_encoder.named_parameters():
                param.requires_grad = learnable
        if self._task_encoder_lstm:
            for name, param in self._task_encoder_lstm.named_parameters():
                param.requires_grad = learnable

    def set_learnable_task_decoder(self, learnable):
        if self._task_decoder:
            for name, param in self._task_decoder.named_parameters():
                param.requires_grad = learnable

    def set_learnable_body_encoder(self, learnable):
        if self._body_encoder:
            for name, param in self._body_encoder.named_parameters():
                param.requires_grad = learnable

    def set_learnable_motor_decoder(self, learnable, free_log_std=True):
        if self._motor_decoder:
            for name, param in self._motor_decoder.named_parameters():
                param.requires_grad = learnable
                if 'log_std' in name:
                    param.requires_grad = free_log_std 

    def set_learnable_motor_decoder_helper(self, learnable):
        if self._motor_decoder_helper:
            for name, param in self._motor_decoder_helper.named_parameters():
                param.requires_grad = learnable

    def set_learnable_future_pred(self, learnable):
        if self._future_pred:
            for name, param in self._future_pred.named_parameters():
                param.requires_grad = learnable

    # @override(nn.Module)
    # def parameters(self):
    #     # {'params': model.base.parameters()},
    #     # {'params': model.classifier.parameters(), 'lr': 1e-3}
    #     param = []
    #     if self._body_encoder:
    #         param.append({'params': self._body_encoder.parameters()})
    #     if self._task_encoder_lstm:
    #         param.append({'params': self._task_encoder_lstm.parameters()})
    #     if self._task_encoder:
    #         param.append({'params': self._task_encoder.parameters()})
    #     if self._motor_decoder:
    #         param.append({'params': self._motor_decoder.parameters()})

    #     # print('==============================')
    #     # print(super().parameters())
    #     # print('==============================')
    #     return param

ModelCatalog.register_custom_model("interaction_net", InteractionPolicy)

ModelCatalog.register_custom_model("fcnn", FullyConnectedPolicy)
ModelCatalog.register_custom_model("task_agnostic_policy_type1", TaskAgnosticPolicyType1)