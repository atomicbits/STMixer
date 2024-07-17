import argparse
import os
import torch.distributed as dist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--transfer", action='store_true')
    parser.add_argument("--no-head", action='store_true')
    parser.add_argument("--use-tfboard", action='store_true')
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    # Print environment variables
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")
    print(f"RANK: {os.environ.get('RANK')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")

    # Initialize the process group
    dist.init_process_group(backend="nccl", init_method="env://")

    print(f"Process with local_rank = {args.local_rank} is running")

if __name__ == "__main__":
    main()