from setuptools import find_packages, setup
from os import path
from pathlib import Path

ROOT = path.abspath(Path(__file__).parent)

# Get requirements from file
requirements = []
with open(path.join(ROOT, "requirements.txt")) as f:
    for line in f.readlines():
        if '#' not in line.strip():
            requirements.append(line.strip())

setup(
    name='htm-sequence-learning',
    author='TaherHabib',
    url='https://github.com/TaherHabib/sequence-learning-model',
    description='Numenta\'s HTM model on Reber Grammar Sequence Learning',
    packages=find_packages(),
    install_requires=requirements,
    version='1.0'
)