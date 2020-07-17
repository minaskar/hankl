import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hankl",
    version="1.1.0",
    author="Minas Karamanis",
    author_email="minaskar@gmail.com",
    description="Lightweight FFTLog for Cosmology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/minaskar/hankl",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    install_requires=["numpy", "scipy"],
    python_requires=">=3.6",
)
