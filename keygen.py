#!/usr/bin/env python3
"""Key generation utility for quantum-resistant media notarizer."""

import click
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from crypto_signing import HybridSigner, KeyManager


@click.command()
@click.option('--output-dir', type=click.Path(path_type=Path), default='.',
              help='Output directory for key files (default: current directory)')
@click.option('--key-name', type=str, default='notarizer',
              help='Base name for key files (default: notarizer)')
@click.option('--use-pqc', is_flag=True, default=False,
              help='Generate post-quantum cryptography keys (requires pqcrypto)')
@click.option('--verbose', '-v', is_flag=True, default=False,
              help='Enable verbose output')
def generate_keys(output_dir: Path, key_name: str, use_pqc: bool, verbose: bool):
    """Generate cryptographic key pairs for signing and verification."""
    
    try:
        if verbose:
            click.echo(f"Generating {'hybrid (classical + PQC)' if use_pqc else 'classical'} key pairs...")
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize signer
        signer = HybridSigner(use_pqc=use_pqc)
        
        # Generate keys
        private_keys, public_keys = signer.generate_keys()
        
        if verbose:
            schemes = list(private_keys.keys())
            click.echo(f"Generated keys for schemes: {', '.join(schemes)}")
        
        # Define file paths
        private_key_path = output_dir / f"{key_name}_private.json"
        public_key_path = output_dir / f"{key_name}_public.json"
        
        # Save keys
        KeyManager.save_keys(
            private_keys, public_keys,
            str(private_key_path), str(public_key_path)
        )
        
        click.echo(f"Keys generated successfully:")
        click.echo(f"  Private key: {private_key_path}")
        click.echo(f"  Public key:  {public_key_path}")
        
        if verbose:
            click.echo(f"\nKey details:")
            for scheme in private_keys.keys():
                key_size = len(private_keys[scheme])
                click.echo(f"  {scheme}: {key_size} bytes")
        
        click.echo(f"\nIMPORTANT: Keep your private key secure and share only the public key!")
        
        return 0
        
    except Exception as e:
        click.echo(f"Error generating keys: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


@click.command()
@click.argument('key_file', type=click.Path(exists=True, path_type=Path))
@click.option('--verbose', '-v', is_flag=True, default=False,
              help='Enable verbose output')
def inspect_keys(key_file: Path, verbose: bool):
    """Inspect cryptographic key file."""
    
    try:
        keys = KeyManager.load_keys(str(key_file))
        
        click.echo(f"Key file: {key_file}")
        click.echo(f"Schemes: {', '.join(keys.keys())}")
        
        if verbose:
            for scheme, key_data in keys.items():
                click.echo(f"\n{scheme}:")
                click.echo(f"  Size: {len(key_data)} bytes")
                click.echo(f"  Type: {'Private' if 'private' in str(key_file).lower() else 'Public'}")
        
        return 0
        
    except Exception as e:
        click.echo(f"Error inspecting keys: {e}", err=True)
        return 1


@click.group()
def cli():
    """Key management utilities for quantum-resistant media notarizer."""
    pass


cli.add_command(generate_keys, name='generate')
cli.add_command(inspect_keys, name='inspect')


if __name__ == '__main__':
    sys.exit(cli())
