#!/usr/bin/env python3
"""
Quick GPU Setup for Live Transcription System
"""

import subprocess
import sys
import platform

def check_gpu():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("⚠️ PyTorch installed but no GPU detected")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def install_pytorch_gpu():
    """Install PyTorch with CUDA support"""
    print("\n📦 Installing PyTorch with a compatible CUDA build (attempting cu121 → cu118 → cpu)...")

    # We try a small set of known-good variants in order. For Python 3.12 Windows
    # machines newer PyTorch builds (2.2+) with cu121 are a good start. If that
    # fails, we try cu118, then fall back to a CPU-only wheel. Installing the
    # correct wheel requires choosing the matching index-url.
    variants = [
        ("2.2.0+cu121", "https://download.pytorch.org/whl/cu121"),
        ("2.2.0+cu118", "https://download.pytorch.org/whl/cu118"),
        ("2.2.0+cpu",   "https://download.pytorch.org/whl/cpu"),
    ]

    for torch_tag, index_url in variants:
        try:
            print(f"→ Trying torch=={torch_tag} from {index_url} ...")
            cmd = [
                sys.executable, "-m", "pip", "install",
                f"torch=={torch_tag}",
                "--index-url", index_url,
            ]
            subprocess.check_call(cmd)
            print(f"✅ Installed torch=={torch_tag}")
            # Optionally install torchvision/torchaudio if needed (commented out
            # because matching tags can be platform specific). Uncomment to enable.
            # subprocess.check_call([sys.executable, "-m", "pip", "install",
            #                        f"torchvision==0.15.2+{cuda_tag}", "--index-url", index_url])
            return True
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Install attempt for {torch_tag} failed: {e}")
        except Exception as e:
            print(f"⚠️ Unexpected error installing {torch_tag}: {e}")

    print("❌ All torch install attempts failed. You can try installing manually using the commands suggested in the README or setup output.")
    return False

def install_requirements():
    """Install all requirements"""
    print("\n📦 Installing requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ All requirements installed!")
        return True
    except Exception as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def main():
    print("="*60)
    print("🚀 LIVE TRANSCRIPTION GPU SETUP")
    print("="*60)
    print(f"Python: {sys.version.split()[0]}")
    print(f"System: {platform.system()}")
    
    # Check GPU
    print("\n🎮 Checking GPU...")
    gpu_available = check_gpu()
    
    if not gpu_available:
        print("\n⚠️ Installing PyTorch with GPU support...")
        if install_pytorch_gpu():
            gpu_available = check_gpu()
    
    # Install other requirements
    print("\n📦 Installing other dependencies...")
    install_requirements()
    
    print("\n"+"="*60)
    if gpu_available:
        print("✅ SETUP COMPLETE - GPU READY!")
    else:
        print("⚠️ SETUP COMPLETE - CPU MODE (slower)")
    print("="*60)
    
    print("\n🚀 To start the server:")
    print("   python live_transcription_server.py")
    print("\n🌐 Then open:")
    print("   live.html in your browser")
    print("\n📋 Features:")
    print("   • Real-time transcription")
    print("   • GPU accelerated (if available)")
    print("   • Hindi, Marathi, English support")
    print("   • Dual speaker tracking")
    print("="*60)

if __name__ == "__main__":
    main()