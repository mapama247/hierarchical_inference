import os
import codecs
import setuptools

HERE = os.path.abspath(os.path.dirname(__file__))
with codecs.open(os.path.join(HERE, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setuptools.setup(
    name='hierarchical_inference',
    version='0.1.0',
    license='MIT',
    url='https://github.com/mapama247/hierarchical_inference',
    description='Python package to perform inference using hierarchical text classifiers.',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords=['NLP', 'text classification', 'hierarchical classification', 'taxonomy learning'],
    author='Marc Pàmies',
    author_email='mpamies247@gmail.com',
    packages=setuptools.find_packages(),
    include_package_data=True,
    platforms=['any'],
    install_requires=[
        'torch',
        'xformers',
        'transformers',
    ],
    classifiers=[
        'Natural Language :: English',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)

