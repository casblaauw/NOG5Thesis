# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="glyc-processing",
    version="1.0.0",
    description="Glycosylation MS data processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Carbyne",
    author_email="15847393+Carbyne@users.noreply.github.com",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent"
    ],
    packages=find_packages(exclude=['tests', 'notebooks']),
    include_package_data=True,
    python_requires=">=3.6",
    # Max versions can be changed/removed, they are only there to prevent problems when automatically installing
    install_requires=["biopython>=1.79,<2",
                      "numpy>=1.19.5,<2",
                      "pandas>=1.1.5,<2",
                      "requests>=2.26.0,<3",
                      "tqdm>=4.62.3,<5",
                      "ipython>=7.27.0,<8"]
)
