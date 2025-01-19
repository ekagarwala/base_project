from setuptools import find_packages, setup

setup(
    name="base_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "xgboost",
        "pyarrow",
        "pytest",
        "scikit-learn",
    ],
    author="Edward Agarwala",
    author_email="edwardagarwala@gmail.com",
    description="A brief description of your project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ekagarwala/base_project",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
)
