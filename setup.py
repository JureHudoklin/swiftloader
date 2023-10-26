from setuptools import setup, find_packages

setup(
    name='SwiftLoader',
    version='0.1',
    package_dir={"": "src"},
    packages=find_packages('src'),
    install_requires=[
        'torch',
        'matplotlib',
        'pycocotools',
    ],
    author='Jure Hudoklin',
    description='A package for loading and processing data in "Swift" format',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
