from setuptools import setup, find_packages

setup(
    # Configuration
    name='leishmaniapp_leishmaniasis_giemsa_macrophages',
    version='2.0.0b1',
    packages=find_packages(),
    package_dir={'': 'src'},

    # Tests
    setup_requires=['pytest-runner'],

    # Entry points
    entry_points={
        'console_scripts': [
            'leishmaniasis_giemsa_macrophages_alef=alef:main',
        ],
    },
)
