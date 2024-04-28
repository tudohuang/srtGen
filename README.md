# srtGen

`srtGen` is a Python module that leverages the Whisper machine learning model to automatically generate SRT (SubRip Subtitle) files from audio. It processes audio files, performs speech-to-text conversion, and formats the output as SRT subtitles, which are widely used in video players and editing software.

## Features

- **Whisper Integration**: Utilizes OpenAI's Whisper model for accurate speech-to-text conversion.
- **SRT Formatting**: Outputs transcription in the standard SRT file format.
- **Language Conversion**: Includes functionality to convert simplified Chinese to traditional Chinese using OpenCC.
- **Customizable Segment Length**: Allows segmentation of the audio input into manageable parts for easier processing.

## Installation

Before installing `srtGen`, ensure you have Python installed on your system. `srtGen` requires Python 3.6 or later.

You can install `srtGen` directly from GitHub using pip:

```bash
pip install git+https://github.com/tudohuang/srtGen.git
```


## Usage

After installation, you can use `srtGen` as a command-line tool or import it into your Python scripts.

### Command Line

To generate SRT files from the command line:

```bash
srt-gen --path /path/to/your/audiofile.wav
```

### In a Python Script

```python
from srtgen import generate_srt

generate_srt("/path/to/your/audiofile.wav")
```

## Dependencies

`srtGen` depends on several third-party libraries:
- Whisper
- PyDub
- NumPy
- OpenCC
- Torch

These dependencies will be installed automatically when installing `srtGen` via pip.

## License

`srtGen` is released under the MIT License. See the LICENSE file for more details.
