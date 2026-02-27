#!/usr/bin/env python3
"""
Validate environment and dependencies for Semantic-Aware GPU Optimization
"""

import sys
import os
import subprocess


def check_python_version():
    """Check Python version >= 3.8"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Requires >= 3.8")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    import_name = import_name or package_name
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "installed")
        print(f"✅ {package_name} ({version})")
        return True
    except ImportError:
        print(f"❌ {package_name} not installed")
        return False


def check_dependencies():
    """Check required dependencies"""
    print("\nChecking dependencies:")
    
    required = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
    ]
    
    all_ok = True
    for package, import_name in required:
        if not check_package(package, import_name):
            all_ok = False
    
    return all_ok


def check_directories():
    """Check if required directories exist"""
    print("\nChecking directories:")
    
    directories = [
        "code",
        "data",
        "results",
        "figures",
        "tests",
        "scripts",
    ]
    
    all_ok = True
    for directory in directories:
        if os.path.isdir(directory):
            print(f"✅ {directory}/")
        else:
            print(f"⚠️  {directory}/ (will be created)")
            all_ok = False
    
    return all_ok


def check_disk_space():
    """Check available disk space"""
    print("\nChecking disk space:")
    
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    
    if free_gb > 0.5:
        print(f"✅ Sufficient disk space ({free_gb:.1f} GB free)")
        return True
    else:
        print(f"❌ Low disk space ({free_gb:.1f} GB free)")
        return False


def check_git():
    """Check if git is available"""
    print("\nChecking git:")
    
    try:
        result = subprocess.run(["git", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        print("⚠️  git not found (optional)")
        return True
    
    return False


def main():
    """Run all checks"""
    print("="*70)
    print("ENVIRONMENT VALIDATION")
    print("="*70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directories", check_directories),
        ("Disk Space", check_disk_space),
        ("Git", check_git),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"⚠️  Error checking {name}: {e}")
            results[name] = False
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")
    
    # Overall status
    all_pass = all(results.values())
    print("\n" + ("="*70))
    if all_pass:
        print("✅ Environment validation PASSED")
        print("You can proceed with running experiments.")
        return 0
    else:
        print("❌ Environment validation FAILED")
        print("Please install missing dependencies:")
        print("  pip install -r code/requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
