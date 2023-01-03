# Seam carving

## About

This is the repository for the final project of the Parallel Programming course. The project requires the base implementation ***Seam carving*** algorithm on both CPU and GPU and makes them robust as possible. It should also be noted that the requirement is to work only on the ***width side*** of a ***pnm*** image (not both sides).

## Contributors

|Name|ID|
|:---:|:---:|
|Võ Thành Nam|19120301|
|Lương Ánh Nguyệt|19120315|

## Algorithm

There are many variants of this algorithm, but in the current project, we use the following:
- Convert the image from RGB to gray.
- Find the energy of all pixels in images using Sobel convolution.
- Find the seam which has the lowest energy.
- Remove the seam from the image.
- Repeat until the image has the expected width.

## How to run

Compile:

```
nvcc support.cu [file-name].cu -o [output]
```

Run on an image:

```
[output] [path-to-image] [n-pixels-to-delete]
```

For example:

```
nvcc support.cu device1.cu -o device1
device1 images/castle.pnm 500
```

## Optimized versions

### Host

### Device version 1

- Use `convert_RGB_to_gray` kernel to convert the RGB image to a gray image.
- Use `conv_sobel` kernel to calculate energy.
- Use `find_seam` kernel and loop from the top row to the bottom row to find all available seams.
- Use `find_min_index` kernel to find the minimum seam.
- Use `remove_seam` kernel to remove the minimum seam.

### Device version 2

From device version 1, we optimized by merging `convert_RGB_to_gray` and `conv_sobel` into one `convert_RGB_to_energy` kernel.

### Device version 3

## Evaluation

### Hardware specification

### Benchmark