from setuptools import setup
from setuptools import find_packages

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content]

setup(
    name="skypkg",
    version="0.0.1",
    author="Astridw94",
    description="End of bootcamp project",
    packages=find_packages(),
    install_requires=requirements
)
