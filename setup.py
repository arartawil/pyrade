"""
PyRADE - Python Rapid Algorithm for Differential Evolution

A high-performance, modular Differential Evolution optimization package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pyrade",
    version="1.0.0",
    author="PyRADE Contributors",
    author_email="",
    description="High-performance, modular Differential Evolution optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pyrade",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
    },
    keywords="optimization, differential evolution, evolutionary algorithms, metaheuristics",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/pyrade/issues",
        "Source": "https://github.com/yourusername/pyrade",
    },
)
