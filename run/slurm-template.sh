#!/bin/bash
# shellcheck disable=SC2206
# THIS FILE IS GENERATED BY AUTOMATION SCRIPT! PLEASE REFER TO ORIGINAL SCRIPT!
# THIS FILE IS A TEMPLATE AND IT SHOULD NOT BE DEPLOYED TO PRODUCTION!

${PARTITION_OPTION}

# Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jungdam@fb.com

#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${OUT_FILE}
#SBATCH --error=${ERR_FILE}
#SBATCH --comment="${COMMENT}"

${GIVEN_NODE}

### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes=${NUM_NODES}
#SBATCH --exclusive
### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gpus-per-task=${NUM_GPUS_PER_NODE}
#SBATCH --mem-per-cpu=4GB
#SBATCH --time ${TIME}

# echo "---------- Stopping the previous ray processes ----------"

ray stop

echo "---------- Loading modules ----------"

# Load modules or your own conda environment here
# module load pytorch/v1.4.0-gpu
# conda activate ${CONDA_ENV}
${LOAD_ENV}

module purge
module load cuda/10.2
module load gcc/7.3.0
# module load cuda/10.1
# module load cudnn/v7.6.5.32-cuda.10.1

unset GLOO_SOCKET_IFNAME
unset NCCL_SOCKET_IFNAME
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE='1'
export RAY_worker_register_timeout_seconds=120

# ===== DO NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING =====
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599
redis_password=$(uuidgen)
export redis_password

echo "---------- pw: $redis_password ----------"

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
nodes_array=($nodes)

echo "----------"
echo "$nodes"
echo "----------"

node_1=${nodes_array[0]}
ips=$(srun --nodes=1 --ntasks=1 -w "$node_1" hostname --all-ip-addresses)
ips_array=( $ips )
ip=${ips_array[0]}
# ip=$(srun --nodes=1 --ntasks=1 -w "$node_1" hostname --all-ip-address) # making redis-address

echo "---------- ip: $ip ----------"

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<< "$ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    ip=${ADDR[1]}
  else
    ip=${ADDR[0]}
  fi
  echo "IPV6 address detected. We split the IPV4 address as $ip"
fi

tmp_dir=${TMP_DIR}
port=${PORT}
ip_head=$ip:$port
export ip_head

echo "---------- ip_head: $ip_head ----------"

# inames = $(srun --nodes=1 --ntasks=1 -w $node_i ip r | grep default | awk '{print $5}')
# inames_array=( $inames )
# iname="${inames_array[0]}"
iname=$(srun --nodes=1 --ntasks=1 -w $node_1 ip r | grep default | awk '{print $5}')

echo "STARTING HEAD at $node_1 / $ip / $iname"
# srun --nodes=1 --ntasks=1 -w "$node_1" ray stop
# sleep 10
srun --nodes=1 --ntasks=1 --export=ALL,GLOO_SOCKET_IFNAME=$iname,NCCL_SOCKET_IFNAME=$iname -w "$node_1" \
  ray start --head --node-ip-address="$ip" --port=$port --redis-password="$redis_password" --temp-dir=$tmp_dir --block &
sleep 30

worker_num=$((SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  # inames = $(srun --nodes=1 --ntasks=1 -w $node_i ip r | grep default | awk '{print $5}')
  # inames_array=( $inames )
  # iname="${inames_array[0]}"
  # ip_i=${ips_array[$i]}
  iname="$(srun --nodes=1 --ntasks=1 -w $node_i ip r | grep default | awk '{print $5}')"
  echo "STARTING WORKER $i at $node_i / $iname"
  # srun --nodes=1 --ntasks=1 -w "$node_i" ray stop
  # sleep 10
  srun --nodes=1 --ntasks=1 --export=ALL,GLOO_SOCKET_IFNAME=$iname,NCCL_SOCKET_IFNAME=$iname, -w "$node_i" \
    ray start --address "$ip_head" --redis-password="$redis_password" --temp-dir=$tmp_dir --block &
  sleep 10
done

sleep 10

# ===== Call your code below =====
${COMMAND_PLACEHOLDER}