from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='easy_exp',
    version="0.1",
    description='Easy Experiment Framework for Machine Learning',
    url="https://github.com/WhuAgent/easy_exp",
    packages=find_packages(),
    install_requires=requirements,
    package_dir={"":"easy_exp"}
)