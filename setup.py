from setuptools import setup, find_packages

setup(
    name='camformat3d',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    license='LICENSE',
    description='An utility library for converting various camera formats to PyTorch3D format.',
    long_description=open('README.md', encoding='utf-8').read(),
    install_requires=[
        "numpy",
        "torch",
        "pytorch3d",
        "plotly",
        "einops",
        "matplotlib"
    ],
)
