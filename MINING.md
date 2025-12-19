# Asynchronous Merge Miner

## Overview

The asynchronous merge miner (`async_merge_miner.py`) is a Python implementation that enables simultaneous mining of Bitcoin, RollPoW, and Tetra-PoW chains using the Stratum protocol. It leverages Python's `asyncio` for efficient, non-blocking I/O operations.

## Features

- **Asynchronous Operation**: Uses async/await patterns for efficient network I/O
- **Stratum Protocol Support**: Connects to standard Bitcoin mining pools
- **Merge Mining**: Mines Bitcoin (BTC), RollPoW, and Tetra-PoW simultaneously
- **Tetra-PoW**: Quantum-temporal proof-of-work with 128-round extended SHA-256 and Ω′ B2 nonlinear feedback
- **Block Persistence**: Saves successfully mined blocks to JSON storage
- **Configurable**: Environment variable support for pool credentials
- **Progress Logging**: Tracks mining statistics and progress

## Requirements

- Python 3.7 or higher (for asyncio support)
- No additional dependencies required (uses standard library)

## Configuration

The miner can be configured using environment variables:

```bash
export MINING_POOL_HOST="solo.ckpool.org"  # Mining pool hostname
export MINING_POOL_PORT="3333"              # Mining pool port
export MINING_ADDRESS="your_btc_address.worker"  # Your Bitcoin address and worker name
export MINING_PASSWORD="x"                  # Mining password (usually "x")
```

If not set, the miner uses default values for demonstration purposes.

## Usage

### Basic Usage

Run the miner directly:

```bash
python3 async_merge_miner.py
```

### With Custom Configuration

```bash
export MINING_ADDRESS="your_bitcoin_address.worker1"
export MINING_POOL_HOST="your.pool.com"
python3 async_merge_miner.py
```

### As a Module

Import and use in your own code:

```python
import asyncio
from async_merge_miner import AsyncStratumClient, AsyncMergeMiner

async def my_mining_app():
    # Create Stratum client
    client = AsyncStratumClient("solo.ckpool.org", 3333, "your_address.worker", "x")
    await client.connect()
    
    # Create miner
    miner = AsyncMergeMiner(client, b"My custom mining message")
    
    # Start mining
    await asyncio.gather(
        client.receive_jobs(),
        miner.mine(iterations=100000)
    )

asyncio.run(my_mining_app())
```

## Output

Mined blocks are saved to `found_blocks/pool_merge_mining_blocks.json` in the following format:

```json
[
    {
        "type": "BTC",
        "nonce": 12345,
        "height": 800000,
        "hash": "0000000000000000000..."
    },
    {
        "type": "RollPoW",
        "nonce": 12346,
        "height": 800000,
        "hash": "0000000000000000000..."
    },
    {
        "type": "Tetra-PoW",
        "nonce": 12347,
        "height": 800000,
        "hash": "0000000000000000000..."
    }
]
```

## Architecture

### Components

1. **AsyncStratumClient**: Handles Stratum protocol communication with mining pools
   - `connect()`: Establishes connection and authorizes with pool
   - `receive_jobs()`: Continuously listens for new mining jobs
   - `submit_share()`: Submits valid shares to the pool

2. **AsyncMergeMiner**: Implements the merge mining logic
   - `mine()`: Main mining loop that tests nonces against BTC, RollPoW, and Tetra-PoW targets
   - `log_progress()`: Periodic progress logging with stats for all three chains

3. **Utility Functions**:
   - `sha256d()`: Double SHA256 hashing for Bitcoin proof-of-work
   - `tetrapow_hash()`: Ω′ Δ18 Tetra-PoW quantum-temporal hash (128-round extended SHA-256)
   - `bits_to_target()`: Converts difficulty bits to target value
   - `save_block_to_file()`: Persists mined blocks to JSON

### Mining Process

1. Connect to Stratum pool and authorize
2. Receive mining job with difficulty target and block template
3. Increment nonce and compute block hash
4. Check if hash meets Bitcoin target → submit share if valid
5. Check if hash meets RollPoW target (relaxed difficulty) → save block if valid
6. Check if hash meets Tetra-PoW target (even more relaxed) → save block if valid
7. Repeat until iteration limit reached

### Tetra-PoW Algorithm

Tetra-PoW is a quantum-temporal proof-of-work algorithm with the following characteristics:

- **128 Rounds**: Extended from SHA-256's 64 rounds for enhanced security
- **Ω′ B2 Nonlinear Feedback**: Quantum feedback term `(A^B^C)^((D<<3)&MASK)^((E>>2)&MASK)`
- **Temporal Mixing**: State registers are updated with enhanced mixing functions
- **Target Difficulty**: Set to `btc_target >> 2` (easier than Bitcoin and RollPoW)

## Logging

The miner logs events at different levels:

- **INFO**: Connection status, job updates, mined blocks, progress
- **DEBUG**: Detailed message exchange with pool
- **ERROR**: Connection failures, parsing errors

## Security Considerations

- Credentials should be configured via environment variables, not hardcoded
- The `found_blocks/` directory is git-ignored to prevent accidental commit
- No private keys or sensitive data are stored by the miner
- Network communication is unencrypted (standard for Stratum)

## Limitations

- This is a demonstration/educational implementation
- Not optimized for actual production mining (CPU mining is not profitable)
- Block header construction is simplified
- RollPoW protocol is a hypothetical example
- No GPU acceleration

## Troubleshooting

### Connection Issues

If you can't connect to the pool:
- Check firewall settings
- Verify pool hostname and port
- Ensure your address format is correct

### No Blocks Found

This is expected! Mining difficulty is extremely high. The miner would need to run for years on a CPU to find a block.

### JSON Errors

If `found_blocks/pool_merge_mining_blocks.json` is corrupted, simply delete it and restart.

## License

This code is part of The Flux Capacitor project and is provided as-is for educational purposes.
