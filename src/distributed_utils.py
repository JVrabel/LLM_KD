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
            for key, value in os.environ.items():
                if any(x in key.lower() for x in ['cuda', 'nccl', 'rank', 'world', 'master', 'local']):
                    logger.info(f"{key}: {value}")
            
            # Get process info
            rank = int(os.environ.get('RANK', '0'))
            world_size = int(os.environ.get('WORLD_SIZE', '1'))
            local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            
            logger.info(f"Process info - Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
            
            # Initialize process group with more explicit error handling
            try:
                init_method = "env://"
                timeout = datetime.timedelta(minutes=30)
                
                # Set CUDA device before init
                torch.cuda.set_device(local_rank)
                
                dist.init_process_group(
                    backend='nccl',
                    init_method=init_method,
                    world_size=world_size,
                    rank=rank,
                    timeout=timeout
                )
                
                # Verify initialization
                if not dist.is_initialized():
                    raise RuntimeError("Failed to initialize process group")
                
                device = torch.device(f"cuda:{local_rank}")
                logger.info(f"Process {rank}/{world_size} using device: {device}")
                logger.info("Distributed setup complete")
                
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