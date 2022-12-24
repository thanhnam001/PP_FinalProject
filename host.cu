#include <stdio.h>
#include <stdint.h>

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
}

void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	
	if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);
	
	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	for (int i = 0; i < width * height; i++)
		fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

	fclose(f);
}

// void writePnm(uchar3 * pixels, int width, int height, char * fileName)
// {
// 	FILE * f = fopen(fileName, "w");
// 	if (f == NULL)
// 	{
// 		printf("Cannot write %s\n", fileName);
// 		exit(EXIT_FAILURE);
// 	}	

// 	fprintf(f, "P3\n%i\n%i\n255\n", width, height); 

// 	for (int i = 0; i < width * height; i++)
// 		fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);
	
// 	fclose(f);
// }

void writePnm(uint8_t * pixels, int numChannels, int width, int height, 
		char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	if (numChannels == 1)
		fprintf(f, "P2\n");
	else if (numChannels == 3)
		fprintf(f, "P3\n");
	else
	{
		fclose(f);
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fprintf(f, "%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height * numChannels; i++)
		fprintf(f, "%hhu\n", pixels[i]);

	fclose(f);
}

char * concatStr(const char * s1, const char * s2){
	char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
	strcpy(result, s1);
	strcat(result, s2);
	return result;
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n");
}

void convert_RGB_to_gray(uchar3* in_pixels, uint8_t* out_pixels, int width, int height){
	for(int r = 0; r < height; r++){
		for(int c = 0; c < width; c++){
			int i = r * width + c;
			uint8_t red = in_pixels[i].x;
			uint8_t green = in_pixels[i].y;
			uint8_t blue = in_pixels[i].z;
			out_pixels[i] = 0.299f * red + 0.587f * green + 0.114f * blue;
		}
	}
}

void conv_sobel(uint8_t* gray_image, uint8_t* energy_image, int width, int height){
	// x-sobel
	int x_sobel[9] = { 1,  0, -1,
					   2,  0, -2,
					   1,  0, -1};
	int y_sobel[9] = { 1,  2,  1,
					   0,  0,  0,
					  -1, -2, -1};
	for(int r = 0; r < height; r++){
		for(int c = 0; c < width; c++){
			int x = 0, y = 0; 
			for(int i = 0; i < 3; i++){
				for(int j = 0; j < 3; j++){
					int rc = min(max(r - 1 + i, 0), height - 1); 
					int cc = min(max(c - 1 + j, 0), width - 1);
					x += gray_image[rc * width + cc] * x_sobel[i * 3 + j];
					y += gray_image[rc * width + cc] * y_sobel[i * 3 + j];
				}
			}
			energy_image[r * width + c] = abs(x) + abs(y);
		}
	}
}

int find_seam(uint8_t* energy_image, uint8_t* back_tracking, int width, int height){
	uint8_t* energy_reduce = (uint8_t*)malloc(width * height);
	int col_start_seam, energy_start_seam = INT8_MAX;
	for(int row = 0; row < height; row++){
		for(int col = 0; col < width; col++){
			int p = row * width + col;
			if(row == 0){
				energy_reduce[p] = energy_image[p];
				back_tracking[p] = 0;
			}else{
				int left = max(col - 1, 0);
				int right = min(col + 1, width - 1);
				energy_reduce[p] = energy_image[p] + energy_reduce[(row - 1) * width + left];
				back_tracking[p] = left;
				if(energy_image[p] + energy_reduce[(row - 1) * width + col] < energy_reduce[p]){
					energy_reduce[p] = energy_image[p] + energy_reduce[(row - 1) * width + col];
					back_tracking[p] = col;
				}
				if(energy_image[p] + energy_reduce[(row - 1) * width + right] < energy_reduce[p]){
					energy_reduce[p] = energy_image[p] + energy_reduce[(row - 1) * width + right];
					back_tracking[p] = right;
				}
				if(row == height - 1 && energy_reduce[p] < energy_start_seam){
					energy_start_seam = energy_reduce[p];
					col_start_seam = col;
				}
			}
		}
	}
	free(energy_reduce);
	return col_start_seam;
}

int main(int argc, char** argv){
    printDeviceInfo();
	int width, height;
	uchar3* original_image;
	readPnm(argv[1], width, height, original_image);
	printf("Image size (width x height): %i x %i\n\n", width, height);

	uint8_t* gray_image = (uint8_t*)malloc(width * height);

	convert_RGB_to_gray(original_image, gray_image, width, height);

	uint8_t* energy_image = (uint8_t*)malloc(width * height);
	conv_sobel(gray_image, energy_image, width, height);

	uint8_t* back_tracking = (uint8_t*)malloc(width * height);
	int col_start_seam = find_seam(energy_image, back_tracking, width, height);

	uint8_t* image_with_seam = (uint8_t*)malloc(width * height * 3);
	for(int row = 0; row < height; row++){
		for(int col = 0; col < width; col++){
			int i = row * width + col;
			image_with_seam[3 * i] = original_image[i].x;
			image_with_seam[3 * i + 1] = original_image[i].y;
			image_with_seam[3 * i + 2] = original_image[i].z;
		}
	}
	for(int row = height - 1; row > -1; row--){
		image_with_seam[3 * (row * width + col_start_seam)] = 255;
		image_with_seam[3 * (row * width + col_start_seam) + 1] = 0;
		image_with_seam[3 * (row * width + col_start_seam) + 2] = 0;
		col_start_seam = back_tracking[row * width + col_start_seam];
	}

	char* file_name_out  = strtok(argv[1], ".");
	writePnm(gray_image, 1, width, height, concatStr(file_name_out,"_gray_host.pnm"));
	writePnm(energy_image, 1, width, height, concatStr(file_name_out,"_energy_host.pnm"));
	writePnm(image_with_seam, 3, width, height, concatStr(file_name_out,"_seam_host.pnm"));

	free(original_image);
	free(gray_image);
	free(energy_image);
	free(image_with_seam);
    return EXIT_SUCCESS;
}