from setuptools import setup, find_packages

setup(
    name='idxein',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[],  # add any dependencies here
    author='Diogo Nardelli Siebert',
    author_email='diogo.siebert@ufsc.br',
    description='A small package for index notation using the Einsten Sum Convention',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/diogosiebert/idxein',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
