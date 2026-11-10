from setuptools import find_packages, setup

setup(
    name="harl",
    version="1.0.0",
    author="PKU-MARL",
    description="HAPPO for five cascade reservoirs, based on PKU-MARL HARL",
    url="https://github.com/PKU-MARL/HARL",
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "pyyaml>=5.3.1",
        "numpy>=1.20,<2",
        "gym>=0.21,<0.27",
        "matplotlib>=3.3",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
