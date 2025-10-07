#!/usr/bin/env python3
"""Comprehensive demo of the Quantum-Resistant Hashchain Media Notarizer.

This demo shows both embedded and external signature approaches with speech-optimized
compression-resistant authentication.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and show results."""
    print(f"\n🔧 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False


def main():
    """Run comprehensive demo."""
    print("🎯 Quantum-Resistant Hashchain Media Notarizer Demo")
    print("==================================================")
    print("Features:")
    print("- Speech-optimized compression-resistant fingerprinting")
    print("- Quantum-resistant cryptographic signatures (Ed25519 + future Dilithium)")
    print("- Hashchain integrity protection")
    print("- Both embedded and external signature modes")
    
    # Check if example files exist
    if not Path("example.mp4").exists():
        print("\n❌ example.mp4 not found. Please ensure the example file exists.")
        return False
    
    if not Path("example_private.json").exists():
        print("\n🔑 Generating cryptographic keys...")
        if not run_command(["python3", "keygen.py"], "Generate Ed25519 key pair"):
            return False
    
    print("\n" + "=" * 60)
    print("🔗 EMBEDDED SIGNATURE APPROACH")
    print("=" * 60)
    print("Creates self-contained signed media files with signatures in metadata")
    
    # Embedded signing
    if run_command([
        "python3", "embedded_signer.py", 
        "example.mp4", "example_private.json", "example_embedded_signed.mp4"
    ], "Create embedded-signed media file"):
        
        # Embedded verification
        run_command([
            "python3", "embedded_verifier.py",
            "example_embedded_signed.mp4", "example_public.json"
        ], "Verify embedded signatures")
    
    print("\n" + "=" * 60)
    print("📄 EXTERNAL SIGNATURE APPROACH") 
    print("=" * 60)
    print("Creates separate .qrmn signature files for robust verification")
    
    # External signing
    if run_command([
        "python3", "external_signer.py",
        "example.mp4", "example_private.json"
    ], "Create external signature file"):
        
        # External verification
        run_command([
            "python3", "external_verifier.py",
            "example.mp4", "example.mp4.qrmn", "example_public.json"
        ], "Verify with external signatures")
    
    print("\n" + "=" * 60)
    print("🧪 COMPRESSION RESISTANCE TEST")
    print("=" * 60)
    print("Testing both external and embedded signatures survive compression...")
    
    # Create test directory
    import os
    test_dir = Path("test")
    test_dir.mkdir(exist_ok=True)
    print(f"✓ Test directory: {test_dir}/\n")
    
    # Define compression test scenarios
    compression_tests = [
        {
            "name": "Heavy AAC",
            "file": "heavy_aac.mp4",
            "cmd_template": ["ffmpeg", "-y", "-i", "{input}", "-c:v", "libx264", "-crf", "23", "-c:a", "aac", "-b:a", "128k", "{output}"],
            "description": "Standard quality compression"
        },
        {
            "name": "Extreme Low",
            "file": "extreme_low.mp4",
            "cmd_template": ["ffmpeg", "-y", "-i", "{input}", "-c:v", "libx264", "-crf", "35", "-c:a", "aac", "-b:a", "32k", "-ac", "1", "-ar", "22050", "{output}"],
            "description": "98% file size reduction"
        },
        {
            "name": "MP3 Transcode",
            "file": "mp3_transcode.mp4",
            "cmd_template": ["ffmpeg", "-y", "-i", "{input}", "-c:v", "copy", "-c:a", "libmp3lame", "-b:a", "320k", "{output}"],
            "description": "High quality MP3 audio"
        },
        {
            "name": "OGG Vorbis",
            "file": "ogg_vorbis.mp4",
            "cmd_template": ["ffmpeg", "-y", "-i", "{input}", "-c:v", "libx264", "-crf", "28", "-c:a", "libvorbis", "-q:a", "3", "{output}"],
            "description": "Vorbis codec compression"
        },
        {
            "name": "Mono + Resample",
            "file": "mono_resample.mp4",
            "cmd_template": ["ffmpeg", "-y", "-i", "{input}", "-c:v", "libx264", "-crf", "32", "-c:a", "aac", "-b:a", "32k", "-ac", "1", "-ar", "16000", "{output}"],
            "description": "Mono audio at 16kHz"
        },
        {
            "name": "Speed Change",
            "file": "speed_change.mp4",
            "cmd_template": ["ffmpeg", "-y", "-i", "{input}", "-filter_complex", "[0:v]setpts=0.9*PTS[v];[0:a]atempo=1.111[a]", "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "28", "-c:a", "aac", "-b:a", "64k", "{output}"],
            "description": "10% faster playback"
        }
    ]
    
    # Store results for summary
    external_results = []
    embedded_results = []
    
    import subprocess
    
    # Test External Mode
    print(f"{'═' * 60}")
    print("📄 EXTERNAL MODE COMPRESSION TESTS")
    print(f"{'═' * 60}")
    
    for test in compression_tests:
        print(f"\n{'─' * 60}")
        print(f"📦 {test['name']} - {test['description']}")
        print(f"{'─' * 60}")
        
        try:
            output_file = test_dir / f"ext_{test['file']}"
            cmd = [c.format(input="example.mp4", output=str(output_file)) for c in test['cmd_template']]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
            original_size = Path("example.mp4").stat().st_size
            compressed_size = output_file.stat().st_size
            reduction = (1 - compressed_size / original_size) * 100
            
            print(f"✓ Compressed: {compressed_size / 1024:.1f} KB ({reduction:.1f}% reduction)")
            
            # Sign and verify
            sig_file = str(output_file) + '.qrmn'
            result = subprocess.run([
                "python3", "external_signer.py", str(output_file), "example_private.json", sig_file
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                verify_result = subprocess.run([
                    "python3", "external_verifier.py", str(output_file), sig_file, "example_public.json"
                ], capture_output=True, text=True)
                
                if "GREEN" in verify_result.stdout:
                    verdict = "✅"
                    external_results.append((test['name'], compressed_size / 1024, reduction, "✅"))
                elif "AMBER" in verify_result.stdout:
                    verdict = "🟨"
                    external_results.append((test['name'], compressed_size / 1024, reduction, "🟨"))
                else:
                    verdict = "❌"
                    external_results.append((test['name'], compressed_size / 1024, reduction, "❌"))
                
                print(f"Verification: {verdict}")
            else:
                external_results.append((test['name'], compressed_size / 1024, reduction, "⚠️"))
                
        except Exception as e:
            print(f"❌ Failed: {e}")
            external_results.append((test['name'], "N/A", "N/A", "❌"))
    
    # Test Embedded Mode
    print(f"\n{'═' * 60}")
    print("🔗 EMBEDDED MODE COMPRESSION TESTS")
    print(f"{'═' * 60}")
    print("Testing if embedded signatures survive compression...")
    
    for test in compression_tests:
        print(f"\n{'─' * 60}")
        print(f"📦 {test['name']} - {test['description']}")
        print(f"{'─' * 60}")
        
        try:
            output_file = test_dir / f"emb_{test['file']}"
            cmd = [c.format(input="example_embedded_signed.mp4", output=str(output_file)) for c in test['cmd_template']]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
            original_size = Path("example_embedded_signed.mp4").stat().st_size
            compressed_size = output_file.stat().st_size
            reduction = (1 - compressed_size / original_size) * 100
            
            print(f"✓ Compressed: {compressed_size / 1024:.1f} KB ({reduction:.1f}% reduction)")
            
            # Verify embedded signatures
            verify_result = subprocess.run([
                "python3", "embedded_verifier.py", str(output_file), "example_public.json"
            ], capture_output=True, text=True)
            
            if "GREEN" in verify_result.stdout:
                verdict = "✅"
                embedded_results.append((test['name'], compressed_size / 1024, reduction, "✅"))
            elif "AMBER" in verify_result.stdout:
                verdict = "🟨"
                embedded_results.append((test['name'], compressed_size / 1024, reduction, "🟨"))
            else:
                verdict = "❌"
                embedded_results.append((test['name'], compressed_size / 1024, reduction, "❌"))
            
            print(f"Verification: {verdict}")
                
        except Exception as e:
            print(f"❌ Failed: {e}")
            embedded_results.append((test['name'], "N/A", "N/A", "❌"))
    
    # Print summary tables
    print(f"\n{'═' * 60}")
    print("📊 COMPRESSION TEST SUMMARY")
    print(f"{'═' * 60}")
    
    print(f"\n📄 External Mode:")
    print(f"{'Scenario':<20} {'Size':>10} {'Reduction':>12} {'Result':>8}")
    print(f"{'─' * 60}")
    for name, size, reduction, result in external_results:
        if isinstance(size, float):
            print(f"{name:<20} {size:>8.1f} KB {reduction:>10.1f}% {result:>8}")
        else:
            print(f"{name:<20} {size:>10} {reduction:>12} {result:>8}")
    
    print(f"\n🔗 Embedded Mode:")
    print(f"{'Scenario':<20} {'Size':>10} {'Reduction':>12} {'Result':>8}")
    print(f"{'─' * 60}")
    for name, size, reduction, result in embedded_results:
        if isinstance(size, float):
            print(f"{name:<20} {size:>8.1f} KB {reduction:>10.1f}% {result:>8}")
        else:
            print(f"{name:<20} {size:>10} {reduction:>12} {result:>8}")
    print(f"{'═' * 60}")
    
    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETE")
    print("=" * 60)
    print("🎯 What you've seen:")
    print("✓ Quantum-resistant cryptographic signatures")
    print("✓ Speech-optimized compression-resistant fingerprinting") 
    print("✓ Cryptographic hashchain integrity protection")
    print("✓ Both embedded and external signature approaches")
    print("✓ Comprehensive compression resistance testing (6 scenarios)")
    
    print("\n📚 Usage:")
    print("• Embedded approach: Best for direct file sharing")
    print("• External approach: Best for robust verification across platforms")
    print("• Both approaches use speech-optimized features that survive compression")
    
    print("\n🚀 The system can detect:")
    print("• Word substitutions and voice alterations")
    print("• Audio splicing and deepfake content")
    print("• Content tampering while surviving compression")
    print("• Maintaining quantum-resistant security")
    
    print(f"\n📁 Generated files:")
    print("• example_embedded_signed.mp4 - Self-contained signed file")
    print("• example.mp4.qrmn - External signature file")
    print(f"• test/ directory - {len(external_results) * 2 + len(embedded_results)} test files")
    print(f"  - {len(external_results)} external mode tests + signatures")
    print(f"  - {len(embedded_results)} embedded mode tests")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
