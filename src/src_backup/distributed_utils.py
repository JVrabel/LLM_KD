import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(cfg):
    """Initialize distributed training if enabled"""
    if not cfg['training'].get('distributed', {}).get('enabled', False):
        return None, 0
    
    if cfg['training']['num_gpus'] > 1:
        # Get SLURM variables
        rank = int(os.environ.get('SLURM_PROCID', '0'))
        world_size = int(os.environ.get('SLURM_NTASKS', '1'))
        local_rank = int(os.environ.get('SLURM_LOCALID', '0'))
        
        # Get master node info from SLURM
        master_addr = os.environ.get('SLURM_LAUNCH_NODE_IPADDR', 'localhost')
        master_port = int(os.environ.get('MASTER_PORT', '29500'))

        # Set up environment variables for distributed training
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)

        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            world_size=world_size,
            rank=rank
        )
        
        # Set device
        torch.cuda.set_device(local_rank)
        
        return {
            'world_size': world_size,
            'rank': rank,
            'local_rank': local_rank
        }, local_rank
        
    return None, 0

def is_distributed(cfg):
    """Check if we're running in distributed mode"""
    return cfg['training'].get('num_gpus', 1) > 1 and cfg['training'].get('distributed', {}).get('enabled', False)

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()