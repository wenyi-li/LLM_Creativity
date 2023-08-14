#!/bin/bash

#- Job parameters

# (TODO)
# Please modify job name

#SBATCH -J vicuna-13b              # The job name
#SBATCH -o ./slurm_output/%x-%j.out        # Write the standard output to file named '<job_name>-<job_number>.out'
#SBATCH -e ./slurm_output/%x-%j.err        # Write the standard error to file named '<job_name>-<job_number>.err'

#- Resources

# (TODO)
# Please modify your requirements

#SBATCH -p r8nv-gpu-hw               # Submit to 'r8nv-gpu-hw' Partitiion
#SBATCH -t 3-00:00:00                # Run for a maximum time of 0 days, 12 hours, 00 mins, 00 secs
#SBATCH --nodes=1                    # Request N nodes
#SBATCH --gres=gpu:8                 # Request M GPU per node
#SBATCH --ntasks-per-node=1          # Request P tasks per node
#SBATCH --cpus-per-task=96           # Request Q core per task; means that P*Q cores per node
#SBATCH --qos=gpu-long             # Request QOS Type
#SBATCH --constraint="IB&A100"

#==========================================================================#
# Please add the SLURM parameter configuration above this horizontal line, #
# and read the README and F.A.Q. at the end of this document.              #
#==========================================================================#

# (TODO)
# Warning: `USER_GPUS_PER_NODE` must be consistent with the above --gres
export USER_GPUS_PER_NODE=8          # <--------------------- Modify it in time!

export USER_NGPUS=$(($USER_GPUS_PER_NODE*$SLURM_JOB_NUM_NODES))
nodelist_h_format=$(scontrol show hostnames $SLURM_JOB_NODELIST | \
    awk -v gpu=$USER_GPUS_PER_NODE '{printf ((NR>1?",":"")$0":%s"), gpu}')


#- Check

if [[ -z $SLURM_NTASKS ]]; then
    echo "SLURM_NTASKS is empty, please check your SBATCH parameter."
    exit -1
fi
if [[ -z $SLURM_NTASKS_PER_NODE ]]; then
    echo "SLURM_NTASKS_PER_NODE is empty, please check your SBATCH parameter."
    exit -1
fi
task_size=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
if [[ $task_size != $SLURM_NTASKS ]]; then
    echo "NTASKS_PER_NODE * NNODE != NNTASK, please check your SBATCH parameter."
    exit -1
fi

if [[ $task_size != $USER_NGPUS ]]; then
    echo "INFO..."
    echo "That's a total of $SLURM_NTASKS tasks, requiring a total of $USER_NGPUS GPUs"
    echo "Becareful whether your program requires \$SLURM_NTASKS or NGPUS"
fi

#- Global Info

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=$(expr 50000 + $(echo -n $SLURM_JOBID | tail -c 4))

### (TODO)
### Warning:
### Sometimes WORLD_SIZE is not the number of tasks, it may be the number of GPUs,
### or even other values which need to be decided according to the actual situation
###
### export WORLD_SIZE=${USER_NGPUS}
### or
### export WORLD_SIZE=${task_size}
###
export WORLD_SIZE=${USER_NGPUS}

#- NCCL Setting

###
### IB here refers to RDMA, not the InfiniBand network in the narrow sense,
### it consists of RDMA over IB network, or RDMA over Converged Ethernet
###
### The NCCL_DEBUG variable controls the debug information that is displayed from NCCL
### INFO - Prints debug information
### export NCCL_DEBUG="INFO"
###

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_HCA="mlx5_0,mlx5_1,mlx5_2,mlx5_3"


#- Log information
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:=        " $SLURM_JOB_NODELIST
echo "Nodelistname:=    " $nodelist_h_format
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= " $SLURM_NTASKS_PER_NODE
echo "Ntasks of jobs:=  " $SLURM_NTASKS
echo "NGPUs of jobs:=   " $USER_NGPUS
echo "MASTER_ADDR:=     " $MASTER_ADDR
echo "MASTER_PORT:=     " $MASTER_PORT
echo "WORLD_SIZE:=      " $WORLD_SIZE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "Job start at $(date "+%Y-%m-%d %H:%M:%S")"
echo "The job is triggered on node:"
echo "$(hostnamectl)"

#- Load environments
source /tools/module_env.sh
module list                       # list modules loaded

#- Tools
module load cluster-tools/v1.0
module load slurm-tools/v1.0
module load cuda-cudnn/11.7-8.5.0
echo "$(df -h | grep -v tmpfs)"
cluster-quota                     # nas quota

#- Job step
# (TODO) Be sure to modify the template.multi-gpus-task.sh file as well.
echo "=============== srun begins =================="
srun bash multi-gpus-task.sh
#- End
echo "Job end at $(date "+%Y-%m-%d %H:%M:%S")"

### README.txt
###
### 1, Limit on the number of cpu cores:
### When set --ntasks-per-node, --cpus-per-task,
### they must meet:
### 1.1, `cpus-per-node_` must be greater than or equal to `DefCpuPerGPU` * `USER_GPUS_PER_NODE`
###      where `cpus-per-node_` := `cpus-per-task` * `ntasks-per-node`.
###      You can view the value of DefCpuPerGPU (usually is 8 or 16)
###      through `scontrol show partition | grep "Partition\|JobDef"`
### 1.2, `cpus-per-node_` needs to be set larger, if enforce-binding is enabled,
###      where `cpus-per-node_` := `cpus-per-task` * `ntasks-per-node`.
###      For 8GPUs&72Cores machines, when each node requires 8 GPUs, 72 is recommended,
###      For 4GPUs&64Cores machines, when each node requires 4 GPUs, 48 and above is recommended,
###      For 8GPUs&128Cores machines, when each node requires 8 GPUs, 96 and above is recommended.
###      But pay attention that `cpu-per-node_` should not exceed the actual number
###      of cpu cores of the host (may be 64, 72, 112, 128, etc. , please use sinfo to view).
###      Another solution is to directly set `--gres-flags=disable-binding`
###
### If your script reported this error:
###    `sbatch: error: Batch job submission failed: Requested node configuration is not available`
### please refer to the above two suggestions.
###
###
### 2, Compute nodes filtering (Recommended)
###
### Without specifying the constraint, any available nodes that meet the requirement will be allocated
### You can specify the characteristics of the compute nodes, and even the names of the compute nodes
###
### # Request a specific list of hosts
### #SBATCH --nodelist=r8a100-d[00-01]
###
### # View available features through `module load slurm-tools && slurm-gpu-info`
### # Eg: Request node type `InfiniBand` and `and` && `IceLake SP Processor`
### #SBATCH --constraint="IB&40G&ILSP
###
###
### 3, CPU-GPU Affinity (Optional)
###
### # Any core can be used with the resources (GPU in this case).
### #SBATCH --gres-flags=disable-binding # mutually exclusive with `enforce-binding`
###
### # Only the identified cores can be allocated with each generic resource,
### # which means that only cpu cores with good affinity can use the GPU.
### #SBATCH --gres-flags=enforce-binding # mutually exclusive with `disable-binding`
###

