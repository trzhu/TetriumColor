from setuptools import setup, find_packages

setup(
    name="TetriumColor",
    version="0.1.0",
    description="A Python library to work in concert with Tetrium",
    author="Jessica Lee",
    packages=find_packages(),
    install_requires=["numpy", "tqdm", "Pillow", "packcircles"],  # Core dependencies
    extras_require={
        "Visualization": ["matplotlib", "scipy"],  # Visualization dependencies
    },
)
