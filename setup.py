import setuptools
import os

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
README = os.path.join(CUR_DIR, "README.md")
with open("README.md", "r") as fd:
    long_description = fd.read()

setuptools.setup(
    name="dssm",
    version="0.1.2",
    description="An industrial-grade implementation of DSSM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Chiang97912/dssm",
    author="Chiang97912",
    author_email="chiang97912@gmail.com",
    packages=["dssm"],
    install_requires=[
        "torch>=1.9.0",
        "nltk>=3.5",
        "numpy>=1.19.5",
        "scikit-learn>=0.21.3"
    ],
    classifiers=(
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ),

    keywords='dssm semantic-retrieval text-matching'
)
