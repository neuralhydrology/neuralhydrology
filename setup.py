from pathlib import Path

from setuptools import setup

# read the description from the README.md
readme_file = Path(__file__).absolute().parent / "README.md"
with readme_file.open("r") as fp:
    long_description = fp.read()

about = {}
with open("neuralhydrology/__about__.py", "r") as fp:
    exec(fp.read(), about)

# TODO: Add Classifier, License and update Authors etc.

setup(name='neuralhydrology',
      version=about["__version__"],
      packages=[
          'neuralhydrology', 'neuralhydrology.data', 'neuralhydrology.utils', 'neuralhydrology.modelzoo',
          'neuralhydrology.training', 'neuralhydrology.evaluation'
      ],
      url='neuralhydrology.readthedocs.io',
      license='',
      author='Frederik Kratzert',
      author_email='f.kratzert@gmail.com',
      description='Library for training deep learning models with environmental focus',
      long_description=long_description,
      long_description_content_type='text/markdown',
      entry_points={
          'console_scripts': [
              'nh-schedule-runs=neuralhydrology.nh_run_scheduler:_main', 'nh-run=neuralhydrology.nh_run:_main',
              'nh-results-ensemble=neuralhydrology.utils.nh_results_ensemble:_main'
          ]
      },
      python_requires='>=3.6',
      install_requires=[
          'matplotlib',
          'numba',
          'numpy',
          'pandas',
          'ruamel.yaml',
          'torch',
          'scipy',
          'tensorboard',
          'tqdm',
          'xarray',
      ],
      classifiers=[
          'Programming Language :: Python :: 3',
          'Operating System :: OS Independent',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Hydrology',
      ],
      keywords='deep learning hydrology lstm neural network streamflow discharge rainfall-runoff')
