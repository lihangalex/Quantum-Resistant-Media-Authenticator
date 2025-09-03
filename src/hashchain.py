"""Cryptographic hash-chain implementation for linking media segments."""

import hashlib
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json


@dataclass
class ChainMessage:
    """Message structure for hash-chain links."""
    window_index: int
    perceptual_hash: bytes
    chain_hash: bytes
    previous_chain_hash: Optional[bytes]
    timestamp: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'window_index': self.window_index,
            'perceptual_hash': self.perceptual_hash.hex(),
            'chain_hash': self.chain_hash.hex(),
            'previous_chain_hash': self.previous_chain_hash.hex() if self.previous_chain_hash else None,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChainMessage':
        """Create from dictionary."""
        return cls(
            window_index=data['window_index'],
            perceptual_hash=bytes.fromhex(data['perceptual_hash']),
            chain_hash=bytes.fromhex(data['chain_hash']),
            previous_chain_hash=bytes.fromhex(data['previous_chain_hash']) if data['previous_chain_hash'] else None,
            timestamp=data['timestamp'],
            metadata=data['metadata']
        )
    
    def serialize_for_signing(self) -> bytes:
        """Serialize message for cryptographic signing."""
        # Create deterministic byte representation
        data = {
            'window_index': self.window_index,
            'perceptual_hash': self.perceptual_hash.hex(),
            'chain_hash': self.chain_hash.hex(),
            'previous_chain_hash': self.previous_chain_hash.hex() if self.previous_chain_hash else None,
            'timestamp': self.timestamp,
            'metadata': json.dumps(self.metadata, sort_keys=True)
        }
        
        # Convert to JSON bytes with sorted keys for deterministic output
        json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return json_str.encode('utf-8')


class HashChain:
    """Cryptographic hash-chain for linking media segments."""
    
    def __init__(self, genesis_data: Optional[bytes] = None):
        """Initialize hash-chain with optional genesis block."""
        self.chain: List[ChainMessage] = []
        self.genesis_hash = self._compute_genesis_hash(genesis_data)
    
    def _compute_genesis_hash(self, genesis_data: Optional[bytes]) -> bytes:
        """Compute genesis hash for chain initialization."""
        if genesis_data is None:
            genesis_data = b"quantum-resistant-media-notarizer-genesis"
        
        return hashlib.sha256(genesis_data).digest()
    
    def _compute_chain_hash(self, window_index: int, perceptual_hash: bytes, 
                          previous_hash: Optional[bytes], timestamp: float,
                          metadata: Dict[str, Any]) -> bytes:
        """Compute hash for chain link."""
        # Combine all elements for chain hash
        hash_input = b''.join([
            window_index.to_bytes(4, 'big'),
            perceptual_hash,
            previous_hash if previous_hash else self.genesis_hash,
            int(timestamp * 1000000).to_bytes(8, 'big'),  # Microsecond precision
            json.dumps(metadata, sort_keys=True).encode('utf-8')
        ])
        
        return hashlib.sha256(hash_input).digest()
    
    def add_window(self, window_index: int, perceptual_hash: bytes, 
                   metadata: Optional[Dict[str, Any]] = None) -> ChainMessage:
        """Add a new window to the hash-chain."""
        if metadata is None:
            metadata = {}
        
        timestamp = time.time()
        previous_hash = self.chain[-1].chain_hash if self.chain else None
        
        chain_hash = self._compute_chain_hash(
            window_index, perceptual_hash, previous_hash, timestamp, metadata
        )
        
        message = ChainMessage(
            window_index=window_index,
            perceptual_hash=perceptual_hash,
            chain_hash=chain_hash,
            previous_chain_hash=previous_hash,
            timestamp=timestamp,
            metadata=metadata
        )
        
        self.chain.append(message)
        return message
    
    def verify_chain_integrity(self) -> bool:
        """Verify the integrity of the entire chain."""
        if not self.chain:
            return True
        
        for i, message in enumerate(self.chain):
            expected_previous = self.chain[i-1].chain_hash if i > 0 else None
            
            if message.previous_chain_hash != expected_previous:
                return False
            
            # Recompute chain hash to verify
            expected_chain_hash = self._compute_chain_hash(
                message.window_index,
                message.perceptual_hash,
                message.previous_chain_hash,
                message.timestamp,
                message.metadata
            )
            
            if message.chain_hash != expected_chain_hash:
                return False
        
        return True
    
    def get_chain_messages(self) -> List[ChainMessage]:
        """Get all chain messages."""
        return self.chain.copy()
    
    def export_chain(self) -> Dict[str, Any]:
        """Export chain to dictionary format."""
        return {
            'genesis_hash': self.genesis_hash.hex(),
            'chain': [msg.to_dict() for msg in self.chain]
        }
    
    @classmethod
    def import_chain(cls, data: Dict[str, Any]) -> 'HashChain':
        """Import chain from dictionary format."""
        chain = cls()
        chain.genesis_hash = bytes.fromhex(data['genesis_hash'])
        chain.chain = [ChainMessage.from_dict(msg_data) for msg_data in data['chain']]
        return chain
