from setuptools import setup, find_packages


setup(name='anomaly_detection',
    version='0.1',
    description='Anomaly Detection',
    url='',
    author='Nina Tuluptceva',
    author_email='nina.tuluptceva@philips.com',
    license='Apache License 2.0',
    packages=find_packages(),
    dependency_links=[],
    install_requires=[
        'numpy', 'scipy', 'tensorboardX',
        'PyYAML', 'scikit-image', 'scikit-learn', 'pandas', 'tqdm'

    ],
    zip_safe=False)
