from setuptools import setup, find_packages


def load_requirements():
    with open("requirements.txt", "r") as f:
        return f.read().split("\n")


setup(
    name="advkit",
    version="0.0.4",
    author="Tianqi Chen",
    description="Adversarial Learning Kit",
    license="MIT",
    url="https://github.com/tqch/advkit",
    packages=find_packages(),
    install_requires=load_requirements()
)
