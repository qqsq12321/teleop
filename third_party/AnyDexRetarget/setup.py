from setuptools import setup, find_packages

setup(
    name="anydexretarget",
    version="0.1.0",
    description="High-precision hand pose retargeting system with adaptive analytical optimization for multiple dexterous hands",
    author="Shiquan Qiu",
    author_email="932851972@qq.com",
    url="https://github.com/qqsq12321/AnyDexRetarget",
    python_requires=">=3.10",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.21.0",
        "nlopt",
        "pyyaml",
    ],
)
