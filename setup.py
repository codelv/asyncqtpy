import os.path
import re

from setuptools import setup

with open("asyncqtpy/__init__.py") as f:
    version = re.search(r"__version__\s+=\s+\"(.*)\"", f.read()).group(1)


desc_path = os.path.join(os.path.dirname(__file__), "README.rst")
with open(desc_path, encoding="utf8") as desc_file:
    long_description = desc_file.read()


setup(
    name="asyncqtpy",
    version=version,
    url="https://github.com/codelv/asyncqtpy",
    author=", ".join(("Gerard Marull-Paretas" "Mark Harviston" "Arve Knudsen")),
    author_email=", ".join(
        (
            "gerard@teslabs.com",
            "mark.harviston@gmail.com",
            "arve.knudsen@gmail.com",
            "frmdstryr@gmail.com",
        )
    ),
    packages=["asyncqtpy"],
    install_requires=["qtpy"],
    license="BSD",
    description="Implementation of the PEP 3156 Event-Loop with Qt using QtPy",
    long_description=long_description,
    python_requires=">=3.9",
    keywords=["Qt", "QtPy", "asyncio"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Environment :: X11 Applications :: Qt",
    ],
)
