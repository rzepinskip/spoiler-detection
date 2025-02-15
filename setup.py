import io
import os
import re

from setuptools import find_packages, setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


setup(
    name="Spoiler detection",
    version="0.1.0",
    url=" ",
    license="MIT",
    author="Paweł Rzepiński",
    author_email=" ",
    description="Spoiler detection models",
    long_description=read("README.md"),
    packages=find_packages(exclude=("tests",)),
    install_requires=["transformers", "wandb", "pytorch-crf", "pytorch-lightning"],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
