#!/usr/bin/env python3
"""
Example usage of the asynchronous merge miner.

This demonstrates how to use the async_merge_miner module
with custom configuration.
"""

import asyncio
import os
from async_merge_miner import AsyncStratumClient, AsyncMergeMiner

async def main():
    """Example mining session with custom configuration."""
    
    # Configure mining parameters
    # In production, use environment variables instead
    pool_host = os.getenv("MINING_POOL_HOST", "solo.ckpool.org")
    pool_port = int(os.getenv("MINING_POOL_PORT", "3333"))
    mining_address = os.getenv("MINING_ADDRESS", "your_btc_address.worker")
    mining_password = os.getenv("MINING_PASSWORD", "x")
    
    print(f"Connecting to {pool_host}:{pool_port}")
    print(f"Mining address: {mining_address}")
    
    # Create mining message (can be customized)
    mining_message = b"Custom merge mining operation"
    
    # Initialize Stratum client
    client = AsyncStratumClient(pool_host, pool_port, mining_address, mining_password)
    
    try:
        # Connect to pool
        await client.connect()
        
        # Create merge miner
        miner = AsyncMergeMiner(client, mining_message)
        
        # Start mining with custom parameters
        # - iterations: number of nonces to try (10000 for quick test)
        # - log_interval: how often to log progress
        await asyncio.gather(
            client.receive_jobs(),      # Listen for jobs from pool
            miner.mine(                 # Mine blocks
                iterations=10000,       # Try 10,000 nonces (adjust as needed)
                log_interval=1000       # Log every 1,000 nonces
            )
        )
    except KeyboardInterrupt:
        print("\nMining interrupted by user")
    except Exception as e:
        print(f"Error during mining: {e}")
    finally:
        # Close connection
        if client.writer:
            client.writer.close()
            await client.writer.wait_closed()

if __name__ == "__main__":
    print("=" * 60)
    print("Asynchronous Merge Miner - Example")
    print("=" * 60)
    print("\nThis is a demonstration of the merge mining capabilities.")
    print("For production use, configure via environment variables:")
    print("  export MINING_ADDRESS='your_address.worker'")
    print("  export MINING_POOL_HOST='your.pool.com'")
    print("\nPress Ctrl+C to stop mining.\n")
    
    # Run the async main function
    asyncio.run(main())
