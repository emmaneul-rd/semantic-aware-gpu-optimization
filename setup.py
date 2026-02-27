from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("code/requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="semantic-gpu-opt",
    version="1.0.0",
    author="Emmanuel Sánchez Pache",
    author_email="emmanuel@salomoncoral.com",
    description="Semantic-Aware GPU Optimization Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/emmaneul-rd/semantic-aware-gpu-optimization",
    project_urls={
        "Bug Tracker": "https://github.com/emmaneul-rd/semantic-aware-gpu-optimization/issues",
        "Documentation": "https://github.com/emmaneul-rd/semantic-aware-gpu-optimization/docs",
        "Paper": "https://Zenodo.org/abs/https://zenodo.org/records/1876502418765024",
        "Research": "https://github.com/emmaneul-rd/semantic-aware-gpu-optimization",
    },
    packages=find_packages(exclude=["tests", "notebooks", "docs", "figures"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.9",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "notebook>=6.0",
            "matplotlib>=3.3",
            "seaborn>=0.11",
        ],
    },
    keywords=[
        "gpu-optimization",
        "semantic-awareness",
        "cache-locality",
        "transformer-inference",
        "energy-efficiency",
        "nvidia-gpu",
        "memory-hierarchy",
    ],
    entry_points={
        "console_scripts": [
            "semantic-gpu-benchmark=code.benchmark_simulation:main",
            "semantic-gpu-validate=scripts.validate_environment:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
