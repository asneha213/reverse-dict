from setuptools import setup

setup(
    name='reversedict',
    version='1.0',
    packages=['src'],
    include_package_data=True,
    install_requires=[
        'flask',
    ],
)
