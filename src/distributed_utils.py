import os
import torch
import torch.distributed as dist
import datetime
import socket
import logging
from torch.nn.parallel import DistributedDataParallel as DDP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_distributed(cfg):
    """Initialize distributed training if enabled"""
    try:
        if not cfg['training'].get('distributed', {}).get('enabled', False):
            logger.info("Distributed training not enabled")
            return None, 0
        
        if cfg['training'].get('num_gpus', 1) > 1:
            # Debug information
            logger.info("=== Distributed Setup Debug Information ===")
            logger.info(f"Hostname: {socket.gethostname()}")
            logger.info(f"CUDA Available: {torch.cuda.is_available()}")
            logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
            
            # Print all environment variables
            logger.info("=== Environment Variables ===")
            env_vars = ['MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK', 'LOCAL_RANK']
            for var in env_vars:
                logger.info(f"{var}: {os.environ.get(var, 'Not set')}")
            
            try:
                # Get process info
                rank = int(os.environ.get('RANK', '0'))
                world_size = int(os.environ.get('WORLD_SIZE', '1'))
                local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            except ValueError as e:
                logger.error("Failed to parse rank/world_size/local_rank from environment")
                raise
            
            logger.info(f"Process info - Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
            
            # Initialize process group with more explicit error handling
            try:
                # Set CUDA device before init
                if torch.cuda.is_available():
                    torch.cuda.set_device(local_rank)
                    logger.info(f"Set CUDA device to: {local_rank}")
                
                # Initialize process group
                logger.info("Initializing process group...")
                dist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    world_size=world_size,
                    rank=rank,
                    timeout=datetime.timedelta(minutes=30)
                )
                
                if not dist.is_initialized():
                    raise RuntimeError("Failed to initialize process group")
                
                logger.info(f"Process group initialized. Is_initialized: {dist.is_initialized()}")
                
                return {
                    'world_size': world_size,
                    'rank': rank,
                    'local_rank': local_rank
                }, local_rank
                
            except Exception as e:
                logger.error(f"Process group initialization failed: {str(e)}", exc_info=True)
                raise
            
    except Exception as e:
        logger.error(f"Error in distributed setup: {str(e)}", exc_info=True)
        raise
        
    return None, 0

def is_distributed(cfg):
    """Check if we're running in distributed mode"""
    return cfg['training'].get('num_gpus', 1) > 1 and cfg['training'].get('distributed', {}).get('enabled', False)

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
            logger.info("Cleaned up distributed process group")
        except Exception as e:
            logger.error(f"Error cleaning up distributed process group: {str(e)}", exc_info=True)