from setuptools import setup, find_packages

setup(
    name='SrtGen',
    version='0.1',
    packages=find_packages(),
    install_requires=['whisper', 'pydub', 'numpy', 'opencc', 'torch'],
    entry_points={
        'console_scripts': [
            'srt-gen = srtgen:main'
        ]
    }
)
