### Example
### sbatch rllib_test_batch.sh

#!/bin/bash
## SLURM scripts have a specific format. 

### Section1: SBATCH directives to specify job configuration

#SBATCH --job-name=rllib_test

## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output=/checkpoint/jungdam/jobs/sample-%i.out
## filename for job standard error output (stderr)
#SBATCH --error=/checkpoint/jungdam/jobs/sample-%i.err

#SBATCH --ntasks=512
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=0
#SBATCH --mem-per-cpu=4GB
#SBATCH --partition=learnfair
#SBATCH --time=71:00:00

driver=$1
spec=$2
ntasks=$3
port=$4

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

echo ---------- Loading pip env ----------

source /private/home/jungdam/Research/venv/multiagent/bin/activate
ray stop

echo ---------- Checking the head IP and others ----------

# Getting the node names
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) 
nodes_array=( $nodes )

## This node will be the head
node1=${nodes_array[0]}

## Get IP of the head node and combine it with a given port
ips=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --all-ip-addresses)
ips_array=( $ips )
ip_prefix=${ips_array[0]}
ip_head=$ip_prefix':'$port

## This is the password that other nodes need to connect to the head
redis_password='fnsofewioeld_dkjdfdsk1234327kk'

echo Head: $node1 $ip_head

## Launch the head
echo ---------- Launching the head node ----------

srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --redis-port=$port --redis-password=$redis_password & # Starting the head
sleep 5

# Must be one less that the total number of nodes
worker_num=$((${#nodes_array[@]} - 1)) 

for ((  i=1; i<=$worker_num; i++ ))
do
    echo ---------- Launching $i th node ----------
    node2=${nodes_array[$i]}
    srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password & # Starting the workers
    sleep 1
done

sleep 5

echo ---------- Staring jobs ----------

python -u $driver --mode train --spec $spec --num_workers $ntasks --ip_head $ip_head --password $redis_password --num_gpus $ngpus # Pass the total number of allocated CPUs

echo ---------- Jobs ended ----------