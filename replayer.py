import sys
import numpy as np

from fairmotion.core.motion import Motion
from fairmotion.ops import conversions, motion, math
from fairmotion.data import bvh
from fairmotion.utils import utils

import argparse
import yaml

from fairmotion.viz.utils import TimeChecker
from fairmotion.viz import glut_viewer
from fairmotion.viz import camera
import render_module as rm
rm.initialize()

import importlib.util
import pybullet as pb
import pybullet_data
import sim_agent
from bullet import bullet_client
from bullet import bullet_render

def arg_parser():
    parser = argparse.ArgumentParser()
    ''' Specification file of the expriment '''
    parser.add_argument("--spec", required=True, type=str)
    return parser

class Replayer(glut_viewer.Viewer):
    def __init__(
        self, 
        config,
        title="replayer", 
        cam=None, 
        size=(1280, 720)):
        
        super().__init__(title, cam, size)
        self.config = config
        self.rm = rm
        self.tex_id_ground = None

        self.pb_client = bullet_client.BulletClient(
            connection_mode=pb.DIRECT, options=' --opengl2')
        self.pb_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        spec = importlib.util.spec_from_file_location(
          "bullet_char_info", self.config['bullet_char_info'])
        bullet_char_info = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bullet_char_info)
        self.bullet_agent = sim_agent.SimAgent(
            pybullet_client=self.pb_client, 
            model_file=self.config['bullet_char_file'], 
            char_info=bullet_char_info,
            ref_scale=1.0,
            self_collision=False,
            kinematic_only=True,
            verbose=False)

if __name__ == "__main__":

    args = arg_parser().parse_args()

    with open(args.spec) as f:
        spec = yaml.load(f, Loader=yaml.FullLoader)

    replayer = Replayer(title=spec['name'], config=spec['env'])
    replayer.run()
