from setuptools import find_packages
from setuptools import setup


setup(
    name="torch_normalization",
    version="0.0.1",
    author="Wangyou Zhang",
    author_email="wyz-97@sjtu.edu.cn",
    description="PyTorch-based implementations of different normalization layers",
    url="https://github.com/Emrys365/torch_normalization",
    license="Apache-2.0",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=["torch>=1.0.0"],
    python_requires=">=3.6",
    tests_require=["torch==2.2.0", "pytest"],
)
