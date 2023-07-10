import collections
import numpy as np

name = "CMU"

''' 
The up direction of the character w.r.t. its root joint.
The up direction in the world frame can be computed by dot(R_root, v_up), 
where R_root is the orientation of the root.
'''
v_up = np.array([0.0, 0.0, 1.0])
''' 
The facing direction of the character w.r.t. its root joint.
The facing direction in the world frame can be computed by dot(R_root, v_face), 
where R_root is the orientation of the root.
'''
v_face = np.array([0.0, -1.0, 0.0])
''' 
The up direction of the world frame, when the character holds its defalult posture (e.g. t-pose).
This information is useful/necessary when comparing a relationship between the character and its environment.
'''
v_up_env = np.array([0.0, 0.0, 1.0])
v_ax1_env = np.array([1.0, 0.0, 0.0])
v_ax2_env = np.array([0.0, 1.0, 0.0])

''' 
Definition of Link/Joint (In our character definition, one joint can only have one link)
'''
Root = -1
Root_geom = 0

''' 
Definition of the root (base) joint
'''
ROOT = Root


''' 
Mapping from joint indicies to names
'''
joint_name = collections.OrderedDict()

joint_name[Root] = "Root"
joint_name[Root_geom] = "Root_geom"

''' 
Mapping from joint names to indicies
'''
joint_idx = collections.OrderedDict()

joint_idx["Root"] = Root
joint_idx["Root_geom"] = Root_geom


''' 
Mapping from character's joint indicies to bvh's joint names.
Some entry could have no mapping (by assigning None).
'''
bvh_map = collections.OrderedDict()

bvh_map[Root] = "Root"


''' 
Mapping from bvh's joint names to character's joint indicies.
Some entry could have no mapping (by assigning None).
'''
bvh_map_inv = collections.OrderedDict()

bvh_map_inv["big_box"] = Root

dof = {
    Root : 6,
    Root_geom : 0
    }

''' 
Definition of PD gains
'''

kp = {}
kd = {}

kp['spd'] = {
    Root : 0
    }

kd['spd'] = {}
for k, v in kp['spd'].items():
    kd['spd'][k] = 0.1 * v
kd['spd'][ROOT] = 0

''' 
Definition of PD gains (tuned for Contrained PD Controller).
"cpd_ratio * kp" and "cpd_ratio * kd" will be used respectively.
'''
cpd_ratio = 0.0002


''' 
Maximum forces that character can generate when PD controller is used.
'''
max_force = {
    Root : 0,
    }

contact_allow_map = {
    Root : False,
    }

joint_weight = {
    Root : 1.0,
    }

sum_joint_weight = 0.0
for key, val in joint_weight.items():
    sum_joint_weight += val
for key, val in joint_weight.items():
    joint_weight[key] /= sum_joint_weight

collison_ignore_pairs = [
    # (Spine, LeftShoulder),
    # (Spine, RightShoulder),
    # (Spine1, LeftShoulder),
    # (Spine1, RightShoulder),
    # (Neck, LeftShoulder),
    # (Neck, RightShoulder),
    # (LowerBack, LeftShoulder),
    # (LowerBack, RightShoulder),
    # (LHipJoint, RHipJoint),
    # (LHipJoint, LowerBack),
    # (RHipJoint, LowerBack),
    # (LHipJoint, Spine),
    # (RHipJoint, Spine),
    # (LeftShoulder, RightShoulder),
    # (Neck, Head),
]

friction_lateral = 0.8
friction_spinning = 0.3
restitution = 0.0