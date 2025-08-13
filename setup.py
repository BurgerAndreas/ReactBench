from setuptools import setup, find_packages

setup(
    name="ReactBench",
    version="0.1.0",
    # packages=find_packages(),
    packages=["ReactBench"],
    entry_points={
        "console_scripts": [
            "reactbench=ReactBench.main:main",
        ],
    },
)
