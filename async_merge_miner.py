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
        # The block height is typically embedded in the coinbase1 field (first bytes)
        coinbase_bytes = bytes.fromhex(coinbase1) if isinstance(coinbase1, str) else coinbase1
        block_height_bytes = coinbase_bytes[:16]  # Usually the first 4 bytes represent the height
        return int.from_bytes(block_height_bytes, byteorder="little")


    async def submit_share(self, job_id: str, nonce: int):
        """
        Submit a valid share.
        """
        submit_data = {
            "method": "mining.submit",
            "params": [self.username, job_id, '00', str(nonce), '000000']
        }
        await self._send(submit_data)
        logging.info(f"Share submitted: Job ID {job_id}, Nonce {nonce}")


# ==================== Asynchronous Merge Miner ====================


class AsyncMergeMiner:
    """
    Asynchronous Merge Miner for Bitcoin and RollPoW.
    """

    def __init__(self, stratum_client: AsyncStratumClient, message: bytes):
        self.stratum_client = stratum_client
        self.message = message
        self.start_time = time.time()
        self.blocks_mined_btc = 0
        self.blocks_mined_rollpow = 0

    def log_progress(self, nonce: int):
        """
        Log periodic progress with nonce and chain statistics.
        """
        elapsed_time = time.time() - self.start_time
        logging.info(f"[Progress] Elapsed Time: {elapsed_time:.2f}s, Nonce: {nonce}")
        logging.info(f"BTC Blocks Found: {self.blocks_mined_btc}, RollPoW Blocks Found: {self.blocks_mined_rollpow}")

    async def mine(self, iterations: int = 100000, log_interval: int = 500):
        """
        Perform asynchronous mining for Bitcoin and RollPoW.
        """
        logging.info("Starting asynchronous merge mining...")
        nonce = 0xA6B7C9FFEECAFEFF

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

            # For simplicity, reuse RollPoW mining logic
            rollpow_target = btc_target << 1  # Assume RollPoW has a slightly looser difficulty
            block_hash_rollpow = sha256d(block_header[::-1])  # Use reversed payload for variation
            if int.from_bytes(block_hash_rollpow, "big") < rollpow_target:
                logging.info(f"RollPoW Block Mined! Nonce: {nonce}, Height: {block_height}, Hash: {block_hash_rollpow.hex()}")
                save_block_to_file(
                    {"type": "RollPoW", "nonce": nonce, "height": block_height, "hash": block_hash_rollpow.hex()}, BLOCKS_FILE
                )
                self.blocks_mined_rollpow += 1

            nonce += 1


# ==================== Main ====================


async def main():
    logging.info("🚀 Launching Asynchronous Merge Miner...")

    # Prepare mining message
    mining_message = b"Asynchronous Bitcoin and RollPoW merge mining"
    logging.info(f"Mining payload: {mining_message.decode()}")

    # Initialize Stratum client for Bitcoin
    btc_client = AsyncStratumClient("solo.ckpool.org", 3333, "33vn9iMkLrj2Kjc2grNjmZPu8WxXSQb8yj.worker", "x")
    await btc_client.connect()

    # Start merge miner
    miner = AsyncMergeMiner(btc_client, mining_message)
    await asyncio.gather(
        btc_client.receive_jobs(),  # Listen for Stratum jobs
        miner.mine(50000),  # Perform asynchronous mining
    )


if __name__ == "__main__":
    asyncio.run(main())
