"""
Setup script for agentic-llm-eval package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="agentic-llm-eval",
    version="0.2.0",
    author="mohamjad",
    description="A comprehensive framework for evaluating agentic LLM behavior",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mohamjad/agentic-llm-eval",
    project_urls={
        "Documentation": "https://github.com/mohamjad/agentic-llm-eval/tree/main/docs",
        "Source": "https://github.com/mohamjad/agentic-llm-eval",
        "Tracker": "https://github.com/mohamjad/agentic-llm-eval/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
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
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "mypy>=1.5.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
)
