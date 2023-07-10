# Usage:
# python slurm-launch.py --exp-name test \
#     --command "rllib train --run PPO --env CartPole-v0"

import argparse
import subprocess
import sys
import time

from pathlib import Path

import os

template_file = Path(__file__).parent / "slurm-template.sh"
TMP_DIR = "${TMP_DIR}"
JOB_NAME = "${JOB_NAME}"
OUT_FILE = "${OUT_FILE}"
ERR_FILE = "${ERR_FILE}"
NUM_NODES = "${NUM_NODES}"
NUM_GPUS_PER_NODE = "${NUM_GPUS_PER_NODE}"
PARTITION_OPTION = "${PARTITION_OPTION}"
COMMAND_PLACEHOLDER = "${COMMAND_PLACEHOLDER}"
GIVEN_NODE = "${GIVEN_NODE}"
LOAD_ENV = "${LOAD_ENV}"
TIME = "${TIME}"
COMMENT = "${COMMENT}"
PORT = "${PORT}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="The job name and path to logging file (exp_name.log).")
    parser.add_argument(
        "--num-nodes",
        "-n",
        type=int,
        default=1,
        help="Number of nodes to use.")
    parser.add_argument(
        "--node",
        "-w",
        type=str,
        help="The specified nodes to use. Same format as the "
        "return of 'sinfo'. Default: ''.")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="Number of GPUs to use in each node. (Default: 0)")
    parser.add_argument(
        "--partition",
        "-p",
        type=str,
    )
    parser.add_argument(
        "--time",
        "-t",
        type=str,
        default="00:10:00"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=15204,
        help="ip port")
    parser.add_argument(
        "--load-env",
        type=str,
        default="",
        help="The script to load your environment ('module load cuda/10.1')")
    parser.add_argument(
        "--command",
        type=str,
        required=True,
        help="The command you wish to execute. For example: "
        " --command 'python test.py'. "
        "Note that the command must be a string.")
    parser.add_argument(
        "--comment",
        type=str,
        default=" ",
        help="comments")
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default="/tmp/ray/",
        help="temporary dir")
    args = parser.parse_args()

    if args.node:
        # assert args.num_nodes == 1
        node_info = "#SBATCH -w {}".format(args.node)
    else:
        node_info = ""

    job_name = "{}_{}".format(args.exp_name,
                              time.strftime("%m%d-%H%M%S", time.localtime()))

    partition_option = "#SBATCH --partition={}".format(
        args.partition) if args.partition else ""

    out_file = os.path.join('data/temp/slurm/', "{}.out".format(job_name))
    err_file = os.path.join('data/temp/slurm/', "{}.err".format(job_name))

    # ===== Modified the template script =====
    with open(template_file, "r") as f:
        text = f.read()
    text = text.replace(TMP_DIR, str(args.tmp_dir))
    text = text.replace(JOB_NAME, job_name)
    text = text.replace(OUT_FILE, out_file)
    text = text.replace(ERR_FILE, err_file)
    text = text.replace(TIME, args.time)
    text = text.replace(NUM_NODES, str(args.num_nodes))
    text = text.replace(NUM_GPUS_PER_NODE, str(args.num_gpus))
    text = text.replace(PARTITION_OPTION, partition_option)
    text = text.replace(COMMAND_PLACEHOLDER, str(args.command))
    text = text.replace(LOAD_ENV, str(args.load_env))
    text = text.replace(GIVEN_NODE, node_info)
    text = text.replace(COMMENT, str(args.comment))
    text = text.replace(PORT, str(args.port))
    text = text.replace(
        "# THIS FILE IS A TEMPLATE AND IT SHOULD NOT BE DEPLOYED TO "
        "PRODUCTION!",
        "# THIS FILE IS MODIFIED AUTOMATICALLY FROM TEMPLATE AND SHOULD BE "
        "RUNNABLE!")

    # ===== Save the script =====
    script_file = os.path.join('data/temp/slurm/', "{}.sh".format(job_name))
    with open(script_file, "w") as f:
        f.write(text)

    # ===== Submit the job =====
    print("Starting to submit job!")
    subprocess.Popen(["sbatch", script_file])
    print(
        "Job submitted! Script/Out/Err files are at: <{}>, <{}>, <{}>".format(
            script_file, out_file, err_file))
    sys.exit(0)