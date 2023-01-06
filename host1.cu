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
	uint32_t col_start_seam, energy_start_seam = UINT32_MAX;
	for(int row = 1; row < height; row++){
		for(int col = 0; col < width; col++){
			int p = row * width + col;
			// if(row != 0){	// for từ row=1 luôn ko cần if???
				uint32_t temp = UINT32_MAX;

				// update - ko cần for
				for(int i = -1; i < 2; i++){
					int upper_col = min(max(col + i, 0), width - 1);
					if(energy_image[(row - 1) * width + upper_col] < temp){
						temp = energy_image[(row - 1) * width + upper_col];
						back_tracking[p] = upper_col;
					}
				}
				energy_image[p] = energy_image[p] + temp;

				if(row == height - 1 && energy_image[p] < energy_start_seam){
					energy_start_seam = energy_image[p];
					col_start_seam = col;
				}
			// }
		}
	}
	return col_start_seam;
}

// update - hàm backtracking đưa ra kết quả 1d-vector v (có size = row) với v[i] = j là ô [i, j] thuộc seam cần xóa

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
	int new_width = width - 1;
	for(int row = height - 1; row > -1; row--){
		for(int col = 0; col < new_width; col++){
			int i = row * width + col;
			if(col < col_start_seam)
				removed_seam[(new_width) * row + col] = original_image[i];
			else
				removed_seam[(new_width) * row + col] = original_image[i + 1];
		}
		col_start_seam = back_tracking[row * width + col_start_seam];
	}
}

void remove_n_seam(uchar3* original_image, uchar3* out_image, int width, int height, int n_seams){
	uchar3* in_image = (uchar3*)malloc(width * height * sizeof(uchar3));
	memcpy(in_image, original_image, sizeof(uchar3) * width * height);
	uchar3* removed_seam = (uchar3*)malloc((width - 1) * height * sizeof(uchar3));
	uint8_t* gray_image = (uint8_t*)malloc(width * height);
	uint32_t* energy_image = (uint32_t*)malloc(width * height * sizeof(uint32_t));
	uint32_t* back_tracking = (uint32_t*)malloc(width * height * sizeof(uint32_t));
	for(int i = 0; i < n_seams; i++){
		convert_RGB_to_gray(in_image, gray_image, width, height);

		conv_sobel(gray_image, energy_image, width, height);

		int col_start_seam = find_seam(energy_image, back_tracking, width, height);

		remove_seam(in_image, removed_seam, width, height, back_tracking, col_start_seam);
		uchar3* temp = in_image;
		in_image = removed_seam;
		removed_seam = temp;
		width -= 1;
	}
	memcpy(out_image, in_image, sizeof(uchar3) * width * height);
	free(gray_image);
	free(energy_image);
	free(back_tracking);
	free(removed_seam);
	free(in_image);
}

int main(int argc, char** argv){
    printDeviceInfo();
	int width, height;
	uchar3* original_image;
	readPnm(argv[1], width, height, original_image);
	char* file_name_out  = strtok(argv[1], ".");
	int n_seams = argc == 3 ? atoi(argv[2]) : 100;
	printf("Image size (width x height): %i x %i\n\n", width, height);
	uchar3* output_image = (uchar3*)malloc((width - n_seams) * height * sizeof(uchar3));
	
	GpuTimer timer; 
    timer.Start();
	remove_n_seam(original_image, output_image, width, height, n_seams);
	timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());

	writePnm(output_image, width - n_seams, height, concatStr(file_name_out,"_host1.pnm"));

	free(original_image);
	free(output_image);
    return EXIT_SUCCESS;
}