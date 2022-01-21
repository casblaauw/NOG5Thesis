import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


def get_requirements(filename):
    requirements = []
    filename = os.path.join(os.path.dirname(__file__), filename)
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        for line in fd:
            line = re.split(r"#(?!egg=)", line.strip())[0]
            if len(line) > 0:
                requirements.append(line)
    return requirements


setup(
    name="nog5",
    version="0.0.1",
    url="https://github.com/Carbyne/NOG5Thesis",
    author="Jakob Stiesmark",
    author_email="carbynegit@gmail.com",
    description="NetOGlyc 5.0 Thesis",
    #long_description=read("../README.rst"),
    packages=find_packages(exclude=("tests",)),
    entry_points={
        "console_scripts": [
            "nog5=nog5.cli:cli"
        ]
    },
    install_requires=get_requirements("./requirements.txt"),
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
    ],
)
