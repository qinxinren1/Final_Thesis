from setuptools import setup, find_packages

setup(
    name="webpage",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'flask',
        'numpy',
        'librosa',
        'madmom',
    ],
) 