### Example
### ./rllib_test_interactive.sh rllib_driver.py data/config/test.yaml project_dir 512 15201 0

#!/bin/bash
## SLURM scripts have a specific format. 

if [ "$#" -lt 7 ]
then
    echo The arguments are insufficient
    exit 1
fi

driver=$1
spec=$2
project_dir=$3
ntasks=$4
port=$5
ngpus=$6
tmp_dir=$7

echo ---------- Remove tmp directory ----------

rm -r $tmp_dir

### Section 2: Setting environment variables for the job
### Remember that all the module command does is set environment
### variables for the software you need to. Here I am assuming I
### going to run something with python.
### You can also set additional environment variables here and
### SLURM will capture all of them for each task

echo ---------- Loading modules ----------

# Start clean
module purge
module load cuda/10.1
module load cudnn/v7.6.5.32-cuda.10.1

# echo ---------- Loading pip env ----------

# source /private/home/jungdam/Research/venv/fairplay/bin/activate

echo ---------- Unset GLOO/NCCL Env Variables ----------

unset GLOO_SOCKET_IFNAME
unset NCCL_SOCKET_IFNAME


nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
## Must be one less that the total number of nodes
worker_num=$((${#nodes_array[@]} - 1)) 

echo ---------- Stopping the previous ray processes ----------

ray stop

# for ((  i=0; i<=$worker_num; i++ ))
# do
# 	node=${nodes_array[$i]}
# 	srun --nodes=1 --ntasks=1 -w $node ray stop
# 	sleep 1
# done
# sleep 10

echo ---------- Checking the head IP and others ----------

## This node will be the head
head_idx=0
head_node=${nodes_array[$head_idx]}

# Get IP of the head node and combine it with a given port
ips=$(srun --nodes=1 --ntasks=1 -w $head_node hostname --all-ip-addresses)
ips_array=( $ips )
ip_prefix=${ips_array[0]}
ip_head=$ip_prefix':'$port

# head_node=${nodes_array[0]}
# head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
# ip_head=$head_node':'$port

## This is the password that other nodes need to connect to the head
redis_password='fnsofewioeld_dkjdfdsk1234327kk'

echo Head: $head_node $ip_head

## The network interface should be specified to let gloo know which one will be used
# echo ---------- Configure network interfaces ----------
# for ((  i=0; i<=$worker_num; i++ ))
# do
#     node=${nodes_array[$i]}
	
	
# 	echo $iname_array
	
# 	# srun --label --nodes=1 --ntasks=1 -w $node2 export GLOO_SOCKET_IFNAME=$iname
# 	# srun --label --nodes=1 --ntasks=1 -w $node2 --export=GLOO_SOCKET_IFNAME=$iname
# 	echo $node2 $iname
# 	sleep 5
# done

## Launch the head
iname=$(srun --nodes=1 --ntasks=1 -w $head_node ip r | grep default | awk '{print $5}')
echo ---------- Launching the head node $head_node / $iname ----------
srun --label --nodes=1 --ntasks=1 --export=ALL,GLOO_SOCKET_IFNAME=$iname,NCCL_SOCKET_IFNAME=$iname -w $head_node ray start --block --head --port=$port --redis-password=$redis_password --temp-dir=$tmp_dir &
sleep 10
## Launch the others
for ((  i=0; i<=$worker_num; i++ ))
do
	node=${nodes_array[$i]}
	# inames=$(srun --nodes=1 --ntasks=1 -w $node ip -o link show | sed -rn '/^[0-9]+: en/{s/.: ([^:]*):.*/\1/p}')
	# inames=$(srun --nodes=1 --ntasks=1 -w $node ip addr show | awk '/inet.*brd/{print $NF}')
	# iname_array=( $inames )
	# iname=${iname_array[0]}
	iname=$(srun --nodes=1 --ntasks=1 -w $node ip r | grep default | awk '{print $5}')
    if [ $i -eq $head_idx ]
	then
		:
	else
		echo ---------- Launching $i th node $node / $iname ----------
	    srun --label --nodes=1 --ntasks=1 --export=ALL,GLOO_SOCKET_IFNAME=$iname,NCCL_SOCKET_IFNAME=$iname -w $node ray start --block --address=$ip_head --redis-password=$redis_password --temp-dir=$tmp_dir & 
	    sleep 5
	fi
done

sleep 30

echo ---------- Staring jobs ----------

if [ "$#" -eq 7 ]
then
    python -u $driver --mode train --spec $spec --project_dir $project_dir --num_workers $ntasks --ip_head $ip_head --password $redis_password --num_gpus $ngpus
elif [ "$#" -eq 8 ]
then
    checkpoint=$8
    echo $checkpoint
    python -u $driver --mode train --spec $spec --project_dir $project_dir --num_workers $ntasks --ip_head $ip_head --password $redis_password --num_gpus $ngpus --checkpoint $checkpoint
fi

echo ---------- Jobs ended ----------