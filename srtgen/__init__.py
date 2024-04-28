import argparse
from .core import generate_srt

def main():
    parser = argparse.ArgumentParser(description="Generate SRT subtitles from audio file.")
    parser.add_argument('--path', type=str, help='Path to the audio file.')
    args = parser.parse_args()
    result_path = generate_srt(args.path)
    print(f"SRT file created at {result_path}")

if __name__ == "__main__":
    main()
