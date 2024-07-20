import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stelpar",
    version="0.1.0",
    author="Matthew J. Fields",
    author_email="fieldsmatthewj@gmail.com",
    description="Stellar parameter estimation and analysis tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mjfields/stelpar",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        "numpy", 
        "emcee>=3.0.0"
    ]
)