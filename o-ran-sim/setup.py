"""Setup script for O-RAN RL Traffic Steering Simulator."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="oran-rl-simulator",
    version="0.1.0",
    author="O-RAN RL Simulator Team",
    description="Reinforcement Learning based Traffic Steering Simulator for O-RAN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/o-ran-sim",
    packages=find_packages(exclude=["tests", "tests.*", "report", "FINAL", "res", "res_test"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
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
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "sionna": [
            "sionna>=0.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "oran-sim-gui=main:main",
            "oran-sim-run=run_all_experiments:main",
        ],
    },
)
