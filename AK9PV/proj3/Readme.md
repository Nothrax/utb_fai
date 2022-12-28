# PV proj3
Program for applying kernel to a given image. Example cuda app.

# Build
mkdir _build
cd _build
cmake ..
make

# Run
Contains -h switch for help, -f switch for source image filepath, -m switch for selecting mode. Modes:
* 1 - blur
* 2 - Edge detection (transformed to grayscale)
* 3 - Edge detection (transformed to grayscale)