#!/usr/bin/env python3
"""
CRAFT Package Setup.

Installation:
    pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="craft",
    version="1.0.0",
    author="CRAFT Authors",
    description="CRAFT - Calibrated Reasoning with Answer-Faithful Traces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ameame1/CRAFT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "isort>=5.12",
            "flake8>=6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "craft-train-sft=craft.run_sft:main",
            "craft-train-grpo=craft.run_grpo:main",
            "craft-eval=craft.run_eval:main",
        ],
    },
)
