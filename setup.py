from setuptools import setup

setup(name='DMPy',
      version=0.1,
      description='DMPy: Automated pipeline for model construction and parametrization of dynamic '
                  'models based on flux-based SBML models.',
      author='Rik van Rosmalen',
      author_email='rikpetervanrosmalen@gmail.com',
      scripts=['DMPy/pipeline.py']
      )
