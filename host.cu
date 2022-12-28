#include "support.h"

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

void conv_sobel(uint8_t* gray_image, uint32_t* energy_image, int width, int height){
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

int find_seam(uint32_t* energy_image, uint32_t* back_tracking, int width, int height){
	uint32_t* energy_reduce = (uint32_t*)malloc(width * height * sizeof(uint32_t));
	uint32_t col_start_seam, energy_start_seam = UINT32_MAX;
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

void highlight_seam(uchar3* original_image, uchar3* image_with_seam, int width, int height, uint32_t* back_tracking, int col_start_seam){
	for(int row = 0; row < height; row++){
		for(int col = 0; col < width; col++){
			int i = row * width + col;
			image_with_seam[i] = original_image[i];
		}
	}
	for(int row = height - 1; row > -1; row--){
		image_with_seam[(row * width + col_start_seam)] = make_uchar3(255, 0, 0);
		col_start_seam = back_tracking[row * width + col_start_seam];
	}
}

void remove_seam(uchar3* original_image, uchar3* removed_seam, int width, int height, uint32_t* back_tracking, int col_start_seam){
	for(int row = height - 1; row > -1; row--){
		for(int col = 0; col < width - 1; col++){
			int i = row * width + col;
			if(col < col_start_seam)
				removed_seam[(width - 1) * row + col] = original_image[i];
			else
				removed_seam[(width - 1) * row + col] = original_image[i + 1];
		}
		col_start_seam = back_tracking[row * width + col_start_seam];
	}
}

void remove_n_seam(uchar3* original_image, uchar3* out_image, int width, int height, int n_seams){
	uchar3* in_image = (uchar3*)malloc(width * height * sizeof(uchar3));
	uchar3* free_later = in_image;
	memcpy(in_image, original_image, sizeof(uchar3) * width * height);
		uchar3* removed_seam = (uchar3*)malloc((width - 1) * height * sizeof(uchar3));
		uint8_t* gray_image = (uint8_t*)malloc(width * height);
		uint32_t* energy_image = (uint32_t*)malloc(width * height * sizeof(uint32_t));
		uint32_t* back_tracking = (uint32_t*)malloc(width * height * sizeof(uint32_t));
	for(int i = 0; i < n_seams; i++){
		printf("%.2f\n", (float)i/n_seams * 100);
		convert_RGB_to_gray(in_image, gray_image, width, height);

		conv_sobel(gray_image, energy_image, width, height);

		int col_start_seam = find_seam(energy_image, back_tracking, width, height);

		remove_seam(in_image, removed_seam, width, height, back_tracking, col_start_seam);
		uchar3* temp = in_image;
		in_image = removed_seam;
		removed_seam = temp;
		// if(i != n_seams - 1)
		// 	free(in_image);	
		width -= 1;
		// printf("removed seam width %i height %i\n",width,height);
		// for(int i=0;i<width*height;i++){
		// 	printf("%hhu %hhu %hhu", in_image[i].x, in_image[i].y, in_image[i].z);
		// 	printf("\n");
		// }
		// printf("\n");
	}
		free(gray_image);
		free(energy_image);
		free(back_tracking);
		free(removed_seam);
	memcpy(out_image, in_image, sizeof(uchar3) * width * height);
	free(in_image);
}

int main(int argc, char** argv){
    printDeviceInfo();
	int width, height;
	uchar3* original_image;
	readPnm(argv[1], width, height, original_image);
	printf("Image size (width x height): %i x %i\n\n", width, height);

	uint8_t* gray_image = (uint8_t*)malloc(width * height);
	convert_RGB_to_gray(original_image, gray_image, width, height);

	uint32_t* energy_image = (uint32_t*)malloc(width * height * sizeof(uint32_t));
	conv_sobel(gray_image, energy_image, width, height);

	uint32_t* back_tracking = (uint32_t*)malloc(width * height * sizeof(uint32_t));
	int col_start_seam = find_seam(energy_image, back_tracking, width, height);

	uchar3* image_with_seam = (uchar3*)malloc(width * height * sizeof(uchar3));
	highlight_seam(original_image, image_with_seam, width, height, back_tracking, col_start_seam);

	int n_seams = 1000;
	uchar3* output_image = (uchar3*)malloc((width - n_seams) * height * sizeof(uchar3));
	remove_n_seam(original_image, output_image, width, height, n_seams);

	char* file_name_out  = strtok(argv[1], ".");
	// writePnm(gray_image, 1, width, height, concatStr(file_name_out,"_gray_host.pnm"));
	// writePnm(energy_image, 1, width, height, concatStr(file_name_out,"_energy_host.pnm"));
	writePnm(image_with_seam, width, height, concatStr(file_name_out,"_seam_host.pnm"));
	writePnm(output_image, width - n_seams, height, concatStr(file_name_out,"_out_host.pnm"));

	// uchar3* output_image = (uchar3*)malloc((width - 1) * height * sizeof(uchar3));
	// remove_seam(original_image, output_image, width, height, back_tracking, col_start_seam);
	// writePnm(image_with_seam, width, height, "seam.pnm");
	// writePnm(output_image, width - 1, height, "out.pnm");

	// printf("success");

	free(original_image);
	free(gray_image);
	free(energy_image);
	free(back_tracking);
	free(image_with_seam);
	free(output_image);
    return EXIT_SUCCESS;
}