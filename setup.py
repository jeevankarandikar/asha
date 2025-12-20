"""Setup configuration for Project Asha."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="asha",
    version="0.5.0",
    author="Jeevan Karandikar",
    description="EMG-to-hand-pose system for prosthetic control and gesture recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "pyrender>=0.1.45",
        "trimesh>=4.0.0",
        "scipy>=1.11.0",
        "mindrove>=1.0.0",
        "pyqtgraph>=0.13.0",
        "PyQt5>=5.15.0",
        "h5py>=3.9.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
