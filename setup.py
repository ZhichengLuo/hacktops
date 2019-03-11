from setuptools import setup

setup(name='hacktops',
      version='0.1',
      description='The funniest joke in the world',
      url='',
      author='sylvain wlodarczyk',
      author_email='swlodarczyk@slb.com',
      license='SLB',
      packages=['hacktops'],
      scripts=['bin/predict-tops'],
      install_requires=[
            'pandas',
            'pyarrow',
            'sklearn'
      ],
      zip_safe=False)