#!/bin/bash

#- Log infomation

node_task_msg="
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Task run on: $(hostname -s), PID: ${SLURM_TASK_PID},
USE GPU ${CUDA_VISIBLE_DEVICES} of this node (GPUs_PER_Node, not PER_Task);
GlobalID : $SLURM_PROCID    of $SLURM_NTASKS,
NodeID   : $SLURM_NODEID    of $SLURM_JOB_NUM_NODES,
LocalID  : $SLURM_LOCALID    of $SLURM_NTASKS_PER_NODE;
GPUs_PER_Task = $USER_NGPUS / $SLURM_NTASKS = $(($USER_NGPUS/$SLURM_NTASKS)),
MASTER_ADDR   = $MASTER_ADDR
MASTER_PORT   = $MASTER_PORT
WORLD_SIZE    = $WORLD_SIZE

$(nvidia-smi --format=csv --query-gpu=name,driver_version,power.limit)
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
"
echo $node_task_msg

#- Important setting, 
#  otherwise it will cause an error of insufficient RDMA resources
ulimit -l unlimited

#- Load environments
source /tools/module_env.sh
##- language
module load python3/3.8.16

##- CUDA
module load cuda-cudnn/11.7-8.5.0

##- virtualenv
# source xxxxx/activate

echo "Task $SLURM_PROCID: "$(module list)              # list modules loaded
echo "Task $SLURM_PROCID: "$(which gcc)
echo "Task $SLURM_PROCID: "$(which python)
echo "Task $SLURM_PROCID: "$(which python3)

#- Warning! Please not change your CUDA_VISIBLE_DEVICES
#- in `.bashrc`, `env.sh`, or your job script
echo "Node $SLURM_NODEID, LocalID $SLURM_LOCALID: Use GPU ${CUDA_VISIBLE_DEVICES}"
#- The CUDA_VISIBLE_DEVICES variable is assigned and specified by SLURM

#- Job step
for i in unusual situation js is im consequences common_problem
do
    python test.py --model vicuna-13b --task $i --prompt_type basic_prompt
    python test.py --model vicuna-13b --task $i --prompt_type instructive_prompt
    python test.py --model vicuna-13b --task $i --prompt_type CoT_prompt
done
#- End
echo "Task $SLURM_PROCID end at $(date "+%Y-%m-%d %H:%M:%S") on $(hostname -s)"
