from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="stofs-utils",
    version="1.0.0",
    author="Mansur Ali Jisan",
    author_email="mansur.jisan@noaa.gov",
    description="Utilities for STOFS3D (Storm Surge and Tide Operational Forecast System)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mansurjisan/stofs_utils",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Oceanography",
        "Topic :: Scientific/Engineering :: Hydrology",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "netCDF4",
        "matplotlib",
        "pyproj",
        "xarray",
    ],
    extras_require={
        "all": [
            "shapely",
            "geopandas",
            "pyshp",
            "pytest",
        ],
    },
)
