#!/usr/bin/env python3
"""
Asynchronous Merge Miner Using Difficulty and Block Height from the Pool
"""

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Persistent blocks storage
BLOCKS_FILE = Path("found_blocks/pool_merge_mining_blocks.json")

# ==================== Utilities ====================


def sha256d(data: bytes) -> bytes:
    """
    Double SHA256 hashing (used for Proof of Work).
    """
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def tetrapow_hash(data: bytes) -> bytes:
    """
    Ω′ Δ18 TETRA-POW Quantum-Temporal Hash (128-round extended SHA-256).
    Enhanced proof-of-work with nonlinear quantum feedback.
    """
    MASK = 0xFFFFFFFF
    
    # Initialize state registers (A–H) with SHA-256 initial values
    A, B, C, D, E, F, G, H = (
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    )
    
    # SHA-256 constants K, extended to 128 rounds
    K = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ] * 2  # Repeat for 128 rounds
    
    # Prepare message schedule W (pad data and create 128 words)
    msg_len = len(data)
    padded = data + b'\x80'
    padded += b'\x00' * ((120 - len(padded)) % 128)
    padded += (msg_len * 8).to_bytes(8, 'big')
    
    W = []
    for i in range(0, len(padded), 4):
        W.append(int.from_bytes(padded[i:i+4], 'big'))
    
    # Extend to 128 words if needed
    while len(W) < 128:
        W.append(0)
    
    # Tetra-PoW: 128 rounds with Ω′ B2 nonlinear feedback
    for i in range(128):
        # Sigma functions
        S1 = ((H >> 6) | (H << 26)) ^ ((H >> 11) | (H << 21)) ^ ((H >> 25) | (H << 7))
        ch = (E & F) ^ (~E & G)
        
        # Ω′ quantum-temporal feedback term
        omega_feedback = ((A ^ B ^ C) ^ ((D << 3) & MASK) ^ ((E >> 2) & MASK)) & MASK
        
        temp1 = (H + S1 + ch + K[i] + W[i % len(W)] + omega_feedback) & MASK
        
        S0 = ((A >> 2) | (A << 30)) ^ ((A >> 13) | (A << 19)) ^ ((A >> 22) | (A << 10))
        maj = (A & B) ^ (A & C) ^ (B & C)
        temp2 = (S0 + maj) & MASK
        
        # Update registers with Tetra-PoW mixing
        H = G
        G = F
        F = E
        E = (D + temp1) & MASK
        D = C
        C = B
        B = A
        A = (temp1 + temp2) & MASK
    
    # Finalize hash (combine state registers)
    final_hash = b''.join([
        (A & MASK).to_bytes(4, 'big'),
        (B & MASK).to_bytes(4, 'big'),
        (C & MASK).to_bytes(4, 'big'),
        (D & MASK).to_bytes(4, 'big'),
        (E & MASK).to_bytes(4, 'big'),
        (F & MASK).to_bytes(4, 'big'),
        (G & MASK).to_bytes(4, 'big'),
        (H & MASK).to_bytes(4, 'big'),
    ])
    
    return final_hash


def bits_to_target(bits: bytes) -> int:
    """
    Convert Stratum pool `bits` to a target value.
    """
    exponent = bits[-1]
    coefficient = int.from_bytes(bits[:-1], byteorder="big")
    target = coefficient * (2 ** (8 * (exponent - 3)))
    return target


def save_block_to_file(block_data: dict, file_path: Path):
    """
    Save mined blocks persistently in a JSON file.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    existing_data = []

    if file_path.exists():
        with open(file_path, "r") as file:
            try:
                existing_data = json.load(file)
            except Exception:
                pass

    existing_data.append(block_data)

    with open(file_path, "w") as file:
        json.dump(existing_data, file, indent=4)

    logging.info(f"Block saved to {file_path}.")


# ==================== Asynchronous Stratum Client ====================


class AsyncStratumClient:
    """
    Asynchronous Stratum Client for Bitcoin Mining.
    """

    def __init__(self, host: str, port: int, username: str, password: str):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.reader = None
        self.writer = None
        self.job_data = None

    async def connect(self):
        """
        Establish connection to the Stratum mining pool.
        """
        logging.info(f"Connecting to Stratum pool at {self.host}:{self.port}...")
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        await self._send({"id": 1, "method": "mining.subscribe", "params": []})
        await self._send({"id": 2, "method": "mining.authorize", "params": [self.username, self.password]})
        logging.info("Connected to Stratum pool.")

    async def _send(self, message: dict):
        """
        Send a JSON-RPC message to the Stratum pool.
        """
        payload = json.dumps(message) + "\n"
        self.writer.write(payload.encode())
        await self.writer.drain()
        logging.debug(f"Sent message: {message}")

    async def receive_jobs(self):
        """
        Continuously listen for new mining jobs.
        """
        while True:
            line = await self.reader.readline()
            if not line:
                continue

            try:
                job = json.loads(line.decode())
                if job.get("method") == "mining.notify":
                    self.job_data = self._parse_mining_notify(job)
                    logging.info(f"New Stratum job received: {self.job_data}")
            except Exception as e:
                logging.error(f"Failed to parse Stratum job: {e}")

    def _parse_mining_notify(self, job: dict):
        """
        Parse a `mining.notify` message from the Stratum pool.
        """
        params = job.get("params", [])
        return {
            "job_id": params[0],
            "prevhash": bytes.fromhex(params[1]),
            "coinbase1": bytes.fromhex(params[2]),
            "coinbase2": bytes.fromhex(params[3]),
            "merkle_branch": [bytes.fromhex(h) for h in params[4]],
            "version": int(params[5], 16),
            "bits": params[6],  # Difficulty in "bits"
            "time": bytes.fromhex(params[7]),
            "block_height": self._extract_block_height(params[2]),
        }

    def _extract_block_height(self, coinbase1: str) -> int:
        """
        Extract the block height from the `coinbase1` field.
        """
        # The block height is typically embedded in the coinbase1 field after the script length
        coinbase_bytes = bytes.fromhex(coinbase1) if isinstance(coinbase1, str) else coinbase1
        # Skip version (4 bytes) and try to extract height from script push data
        # This is a simplified extraction - actual parsing may vary by pool
        if len(coinbase_bytes) >= 8:
            block_height_bytes = coinbase_bytes[4:8]  # Extract 4 bytes after version
            return int.from_bytes(block_height_bytes, byteorder="little")
        return 0  # Return 0 if unable to extract


    async def submit_share(self, job_id: str, nonce: int):
        """
        Submit a valid share.
        """
        submit_data = {
            "method": "mining.submit",
            "params": [self.username, job_id, '00', f'{nonce:08x}', '000000']
        }
        await self._send(submit_data)
        logging.info(f"Share submitted: Job ID {job_id}, Nonce {nonce}")


# ==================== Asynchronous Merge Miner ====================


class AsyncMergeMiner:
    """
    Asynchronous Merge Miner for Bitcoin, RollPoW, and Tetra-PoW.
    """

    def __init__(self, stratum_client: AsyncStratumClient, message: bytes):
        self.stratum_client = stratum_client
        self.message = message
        self.start_time = time.time()
        self.blocks_mined_btc = 0
        self.blocks_mined_rollpow = 0
        self.blocks_mined_tetrapow = 0

    def log_progress(self, nonce: int):
        """
        Log periodic progress with nonce and chain statistics.
        """
        elapsed_time = time.time() - self.start_time
        logging.info(f"[Progress] Elapsed Time: {elapsed_time:.2f}s, Nonce: {nonce}")
        logging.info(f"BTC Blocks Found: {self.blocks_mined_btc}, RollPoW Blocks Found: {self.blocks_mined_rollpow}, Tetra-PoW Blocks Found: {self.blocks_mined_tetrapow}")

    async def mine(self, iterations: int = 100000, log_interval: int = 500):
        """
        Perform asynchronous mining for Bitcoin, RollPoW, and Tetra-PoW.
        """
        logging.info("Starting asynchronous merge mining (BTC + RollPoW + Tetra-PoW)...")
        nonce = 0

        while nonce < iterations:
            # Periodic progress logs
            if nonce % log_interval == 0:
                self.log_progress(nonce)

            # Ensure valid Stratum job data is available
            if not self.stratum_client.job_data:
                await asyncio.sleep(0.1)  # Wait for a job to be received
                continue

            # Extract block height and target from the pool-provided job data
            job = self.stratum_client.job_data
            bits = bytes.fromhex(job["bits"])
            btc_target = bits_to_target(bits)
            block_height = job["block_height"]

            # Create the block header for Bitcoin
            block_header = self.message + nonce.to_bytes(4, "little")
            block_hash_btc = sha256d(block_header)

            # Bitcoin mining validation
            if int.from_bytes(block_hash_btc, "big") < btc_target:
                await self.stratum_client.submit_share(job["job_id"], nonce)
                logging.info(f"Bitcoin Block Mined! Nonce: {nonce}, Height: {block_height}, Hash: {block_hash_btc.hex()}")
                save_block_to_file(
                    {"type": "BTC", "nonce": nonce, "height": block_height, "hash": block_hash_btc.hex()}, BLOCKS_FILE
                )
                self.blocks_mined_btc += 1

            # RollPoW mining logic
            rollpow_target = btc_target << 1  # RollPoW has easier difficulty (higher target value)
            block_hash_rollpow = sha256d(block_header[::-1])  # Use reversed payload for variation
            if int.from_bytes(block_hash_rollpow, "big") < rollpow_target:
                logging.info(f"RollPoW Block Mined! Nonce: {nonce}, Height: {block_height}, Hash: {block_hash_rollpow.hex()}")
                save_block_to_file(
                    {"type": "RollPoW", "nonce": nonce, "height": block_height, "hash": block_hash_rollpow.hex()}, BLOCKS_FILE
                )
                self.blocks_mined_rollpow += 1

            # Tetra-PoW mining logic (quantum-temporal 128-round hash)
            tetrapow_target = btc_target << 2  # Tetra-PoW has even easier difficulty (higher target value)
            block_hash_tetrapow = tetrapow_hash(block_header)
            if int.from_bytes(block_hash_tetrapow, "big") < tetrapow_target:
                logging.info(f"Tetra-PoW Block Mined! Nonce: {nonce}, Height: {block_height}, Hash: {block_hash_tetrapow.hex()}")
                save_block_to_file(
                    {"type": "Tetra-PoW", "nonce": nonce, "height": block_height, "hash": block_hash_tetrapow.hex()}, BLOCKS_FILE
                )
                self.blocks_mined_tetrapow += 1

            nonce += 1


# ==================== Main ====================


async def main():
    import os
    
    logging.info("🚀 Launching Asynchronous Merge Miner (BTC + RollPoW + Tetra-PoW)...")

    # Prepare mining message
    mining_message = b"Asynchronous Bitcoin, RollPoW, and Tetra-PoW merge mining"
    logging.info(f"Mining payload: {mining_message.decode()}")

    # Get mining configuration from environment or use defaults
    pool_host = os.getenv("MINING_POOL_HOST", "solo.ckpool.org")
    pool_port = int(os.getenv("MINING_POOL_PORT", "3333"))
    mining_address = os.getenv("MINING_ADDRESS", "33vn9iMkLrj2Kjc2grNjmZPu8WxXSQb8yj.worker")
    mining_password = os.getenv("MINING_PASSWORD", "x")

    # Initialize Stratum client for Bitcoin
    btc_client = AsyncStratumClient(pool_host, pool_port, mining_address, mining_password)
    await btc_client.connect()

    # Start merge miner
    miner = AsyncMergeMiner(btc_client, mining_message)
    await asyncio.gather(
        btc_client.receive_jobs(),  # Listen for Stratum jobs
        miner.mine(50000),  # Perform asynchronous mining
    )


if __name__ == "__main__":
    asyncio.run(main())
