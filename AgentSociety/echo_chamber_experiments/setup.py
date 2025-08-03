#!/usr/bin/env python3
"""
Setup script for Echo Chamber Experiments
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    # Fallback requirements if file doesn't exist
    requirements = [
        "numpy>=1.20.0",
        "matplotlib>=3.5.0", 
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "networkx>=2.6.0",
        "plotly>=5.0.0",
        "click>=8.0.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.62.0",
        "scipy>=1.7.0"
    ]

setup(
    name="echo-chamber-experiments",
    version="1.0.0",
    author="Echo Chamber Research Team",
    author_email="research@example.com",
    description="A framework for studying belief propagation and echo chamber formation in AI agent societies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/echo-chamber-experiments",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "jupyter": ["jupyter>=1.0.0", "ipywidgets>=7.6.0"],
        "advanced-viz": ["bokeh>=2.4.0", "dash>=2.0.0"],
        "dev": ["pytest>=6.0", "black>=21.0", "flake8>=3.8"]
    },
    entry_points={
        "console_scripts": [
            "echo-chamber=run_experiment:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "echo_chamber_experiments": [
            "configs/*.json",
            "examples/*.py",
            "README.md"
        ]
    },
    zip_safe=False,
)