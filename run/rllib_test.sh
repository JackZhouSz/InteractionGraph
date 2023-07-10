#!/bin/bash

worker_num=1 # Must be one less that the total number of nodes

# module load Langs/Python/3.6.9 # This will vary depending on your environment
# source /private/home/jungdam/Research/venv/multiagent/bin/activate

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node1=${nodes_array[0]}

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
suffix=':6379'
ip_head=$ip_prefix$suffix
# redis_password=$(uuidgen)
redis_password=jungdam_faircluster_test_012345

export ip_head # Exporting for latter access by rllib_trainer_test.py

srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --redis-port=6379 --redis-password=$redis_password & # Starting the head
sleep 5

for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password & # Starting the workers
  sleep 5
done

python -u rllib_trainer_test.py $redis_password 160 # Pass the total number of allocated CPUs
