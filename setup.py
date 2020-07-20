import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="hfr",
    version="1.0.0",
    description="Hash function representations",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/sgarg87/HFR",
    author="Sahil Garg",
    author_email="sahil.garg.cs@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["hash_sentences"],
    include_package_data=True,
    install_requires=["numpy", "nltk", "keras", "Cython", "scipy", "scikit_learn"],
    entry_points={
        "console_scripts": [
            "realpython=hash_sentences.__main__:main",
        ]
    },
)
