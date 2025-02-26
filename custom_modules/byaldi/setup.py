from setuptools import setup, find_packages

setup(
    name="custom_byaldi",
    version="0.1.0",
    packages=find_packages(include=["."]),
    install_requires=[
        "torch",
        "srsly",
        "byaldi",  # The base byaldi package must be installed first
    ],
) 