from setuptools import setup, find_packages

setup(
    name = 'anomaly_detection',
    version = '0.0.1',
    author = 'kpapdac',
    author_email = 'kpapdac@gmail.com',
    description = ('Anomaly detection algorithms'),
    license = '',
    keywords = 'outliers anomaly clustering',
    url = 'https://github.com/kpapdac/anomaly_detection',
    packages = find_packages(),
    zip_safe= False,
    classifiers = [
    ],
    include_package_data = False,
    install_requires = [
      'numpy',
      'matplotlib',
    ],
    tests_require = [
    ],
    extras_require = {
        'docs': [
          'Sphinx',
        ],
    },
)