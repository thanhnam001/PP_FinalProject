#include "support.h"

#define FILTER_WIDTH 3
__constant__ short int dc_x_sobel[FILTER_WIDTH * FILTER_WIDTH];
__constant__ short int dc_y_sobel[FILTER_WIDTH * FILTER_WIDTH];

__global__ void convert_RGB_to_energy(uchar3* original_image, uint32_t* energy_image, int width, int height){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ uchar3 shared_image[];

    if(row < height && col < width){
        // Copy to shared memory
        int s_size = blockDim.x + 2;
        int s_row = threadIdx.y + 1;
        int s_col = threadIdx.x + 1;

        shared_image[s_row * s_size + s_col] = original_image[row * width + col];

        // Padding edge cells
        int left = max(0, col - 1);
		int right = min(col + blockDim.x, width - 1);
		int top = max(0, row - 1);
		int bottom = min(row + blockDim.y, height - 1);

        if (threadIdx.x == 0) {
            // left and right edge in s_row
            shared_image[s_row * s_size] = original_image[row * width + left];
			shared_image[s_row * s_size + blockDim.x + 1] = original_image[row * width + right];
            if (threadIdx.y == 0) {
                // 4 corners of padding
                shared_image[0] = original_image[top * width + left];
				shared_image[blockDim.x + 1] = original_image[top * width + right];
				shared_image[(blockDim.y + 1) * s_size] = original_image[bottom * width + left];
				shared_image[(blockDim.y + 1) * s_size + blockDim.x + 1] = original_image[bottom * width + right];
            }
        }
        if (threadIdx.y == 0) {
            // top and bottom edge in s_col
            shared_image[threadIdx.x + 1] = original_image[top * width + col];
			shared_image[(blockDim.y + 1) * s_size + threadIdx.x + 1] = original_image[bottom * width + col];
        }

        __syncthreads();

        // Compute energy
        int x = 0, y = 0;
        for(int r = 0; r < FILTER_WIDTH; r++){
            for(int c = 0; c < FILTER_WIDTH; c++){
                uchar3 pixel = shared_image[(threadIdx.y + r) * s_size + threadIdx.x + c];
                uint32_t gray_pixel = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
                x += gray_pixel * dc_x_sobel[r * FILTER_WIDTH + c];
                y += gray_pixel * dc_y_sobel[r * FILTER_WIDTH + c];
            }
        }
        energy_image[row * width + col] = abs(x) + abs(y);
    }
}

__global__ void find_seam(uint32_t* energy_image, uint32_t* back_tracking, int width, int height, int row){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t temp = UINT32_MAX;
    if(col < width){
        int p = row * width + col;
        for(int i = -1; i < 2; i++){
            int upper_col = min(max(col + i, 0), width - 1);
            if(energy_image[(row - 1) * width + upper_col] < temp){
                temp = energy_image[(row - 1) * width + upper_col];
                back_tracking[p] = upper_col;
            }
        }
        energy_image[p] += temp;
    }
}

__device__ void warp_reduce(volatile uint32_t* s_data, volatile uint32_t* s_min_indices, unsigned int tid){
    if(blockDim.x >= 64){
        if(s_data[tid + 32] < s_data[tid]){
            s_data[tid] = s_data[tid + 32];
            s_min_indices[tid] = s_min_indices[tid + 32];
        } 
        else if(s_data[tid + 32] == s_data[tid])
            s_min_indices[tid] = min(s_min_indices[tid], s_min_indices[tid + 32]);
    }
    if(blockDim.x >= 32){
        if(s_data[tid + 16] < s_data[tid]){
            s_data[tid] = s_data[tid + 16];
            s_min_indices[tid] = s_min_indices[tid + 16];
        } 
        else if(s_data[tid + 16] == s_data[tid])
            s_min_indices[tid] = min(s_min_indices[tid], s_min_indices[tid + 16]);
    }
    if(blockDim.x >= 16){
        if(s_data[tid + 8] < s_data[tid]){
            s_data[tid] = s_data[tid + 8];
            s_min_indices[tid] = s_min_indices[tid + 8];
        } 
        else if(s_data[tid + 8] == s_data[tid])
            s_min_indices[tid] = min(s_min_indices[tid], s_min_indices[tid + 8]);
    }
    if(blockDim.x >= 8){
        if(s_data[tid + 4] < s_data[tid]){
            s_data[tid] = s_data[tid + 4];
            s_min_indices[tid] = s_min_indices[tid + 4];
        } 
        else if(s_data[tid + 4] == s_data[tid])
            s_min_indices[tid] = min(s_min_indices[tid], s_min_indices[tid + 4]);
    }
    if(blockDim.x >= 4){
        if(s_data[tid + 2] < s_data[tid]){
            s_data[tid] = s_data[tid + 2];
            s_min_indices[tid] = s_min_indices[tid + 2];
        } 
        else if(s_data[tid + 2] == s_data[tid])
            s_min_indices[tid] = min(s_min_indices[tid], s_min_indices[tid + 2]);
    }
    if(blockDim.x >= 2){
        if(s_data[tid + 1] < s_data[tid]){
            s_data[tid] = s_data[tid + 1];
            s_min_indices[tid] = s_min_indices[tid + 1];
        } 
        else if(s_data[tid + 1] == s_data[tid])
            s_min_indices[tid] = min(s_min_indices[tid], s_min_indices[tid + 1]);
    }
}

__global__ void find_min_index(uint32_t* last_row, uint32_t* minn, int n, uint32_t* min_indices){
    extern __shared__ uint32_t s[];
    uint32_t *s_data = (uint32_t*)s;
    uint32_t *s_min_indices = (uint32_t*)&s_data[blockDim.x]; // == data[i] = data + i (+ 4)
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int grid_size = blockDim.x * 2 * gridDim.x;
    s_min_indices[threadIdx.x] = i < n ? i : 0;
    s_min_indices[threadIdx.x + blockDim.x] = i + blockDim.x < n ? i + blockDim.x : 0;
    __syncthreads();
    if(i >= n)
        s_data[threadIdx.x] = UINT32_MAX;
    while(i < n){
        s_data[threadIdx.x] = last_row[i];
        if(i + blockDim.x < n){
            if (last_row[i] > last_row[i + blockDim.x]){
                s_data[threadIdx.x] = last_row[i + blockDim.x];
                s_min_indices[threadIdx.x] = s_min_indices[threadIdx.x + blockDim.x];
            }
        }
        i += grid_size;
    }
    __syncthreads();
    if(blockDim.x >= 512){
        if(threadIdx.x < 256){
            if (s_data[threadIdx.x + 256] < s_data[threadIdx.x]){
                s_data[threadIdx.x] = s_data[threadIdx.x + 256];
                s_min_indices[threadIdx.x] = s_min_indices[threadIdx.x + 256];
            } 
            else if (s_data[threadIdx.x + 256] == s_data[threadIdx.x])
                s_min_indices[threadIdx.x] = min(s_min_indices[threadIdx.x], s_min_indices[threadIdx.x + 256]);
        }
        __syncthreads();
    }
    if(blockDim.x >= 256){
        if(threadIdx.x < 128){
            if (s_data[threadIdx.x + 128] < s_data[threadIdx.x]){
                s_data[threadIdx.x] = s_data[threadIdx.x + 128];
                s_min_indices[threadIdx.x] = s_min_indices[threadIdx.x + 128];
            } 
            else if (s_data[threadIdx.x + 128] == s_data[threadIdx.x])
                s_min_indices[threadIdx.x] = min(s_min_indices[threadIdx.x], s_min_indices[threadIdx.x + 128]);
        }
        __syncthreads();
    }
    if(blockDim.x >= 128){
        if(threadIdx.x < 64){
            if (s_data[threadIdx.x + 64] < s_data[threadIdx.x]){
                s_data[threadIdx.x] = s_data[threadIdx.x + 64];
                s_min_indices[threadIdx.x] = s_min_indices[threadIdx.x + 64];
            } 
            else if (s_data[threadIdx.x + 64] == s_data[threadIdx.x])
                s_min_indices[threadIdx.x] = min(s_min_indices[threadIdx.x], s_min_indices[threadIdx.x + 64]);
        }
        __syncthreads();
    }
    if(threadIdx.x < 32)
        warp_reduce(s_data, s_min_indices, threadIdx.x);
    if(threadIdx.x == 0){
        minn[blockIdx.x] = s_data[0];
        min_indices[blockIdx.x] = s_min_indices[0];
    }
}

__global__ void remove_seam(uchar3* in_image, uchar3* out_image, int width, int height, uint32_t* seam){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if(row < height && col < width - 1){
        if(col < seam[row])
            out_image[(width - 1) * row + col] = in_image[row * width + col];
        else
            out_image[(width - 1) * row + col] = in_image[row * width + col + 1];
    }
}
void remove_n_seam(uchar3* original_image, uchar3* out_image, int width, int height, int n_seams){
    short int x_sobel[FILTER_WIDTH * FILTER_WIDTH] = { 1,  0, -1,
					                                   2,  0, -2,
					                                   1,  0, -1};
	short int y_sobel[FILTER_WIDTH * FILTER_WIDTH] = { 1,  2,  1,
                                                       0,  0,  0,
                                                      -1, -2, -1};
    CHECK(cudaMemcpyToSymbol(dc_x_sobel, x_sobel, FILTER_WIDTH * FILTER_WIDTH * sizeof(short int)));
    CHECK(cudaMemcpyToSymbol(dc_y_sobel, y_sobel, FILTER_WIDTH * FILTER_WIDTH * sizeof(short int)));
    uchar3 *d_original_image;
    uint32_t *d_energy_image;
    uint32_t *d_back_tracking;
    uint32_t *d_seam;
    uchar3 * d_output_image;
    uint32_t *d_minn, *d_min_indices;
    size_t n_bytes_uchar3 = width * height * sizeof(uchar3);
    size_t n_bytes_uint32t = width * height * sizeof(uint32_t);
    size_t n_bytes_row = width * sizeof(uint32_t);
    size_t n_bytes_height = height * sizeof(uint32_t);
    CHECK(cudaMalloc(&d_original_image, n_bytes_uchar3));
    CHECK(cudaMalloc(&d_energy_image, n_bytes_uint32t));
    CHECK(cudaMalloc(&d_back_tracking, n_bytes_uint32t));
    CHECK(cudaMalloc(&d_seam, n_bytes_height));
    CHECK(cudaMalloc(&d_output_image, n_bytes_uchar3));
    CHECK(cudaMemcpy(d_original_image, original_image, n_bytes_uchar3, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_minn, n_bytes_row));
    CHECK(cudaMalloc(&d_min_indices, n_bytes_row));
    uint32_t* seam = (uint32_t*)malloc(n_bytes_height);
    uint32_t* back_tracking = (uint32_t*)malloc(n_bytes_uint32t);
    dim3 block_size2d(32,32);
    dim3 grid_size2d((width - 1) / block_size2d.x + 1, (height - 1) / block_size2d.y + 1);
    dim3 block_size1d(512);
    dim3 grid_size1d((width - 1) / block_size1d.x + 1);
    uint32_t col_start_seam;
    dim3 grid_size1d2((width - 1) / block_size1d.x / 2+ 1);
    uint32_t *minn = (uint32_t*)malloc(grid_size1d2.x * sizeof(uint32_t));
    uint32_t *min_indices = (uint32_t*)malloc(grid_size1d2.x * sizeof(uint32_t));
    uint32_t* accum_last_row = (uint32_t*)malloc(width * sizeof(uint32_t));   
    for(int i = 0; i < n_seams; i++){
        // Convert RGB to gray and calculate energy
        convert_RGB_to_energy<<<grid_size2d, block_size2d, (block_size2d.x + 2) * (block_size2d.y + 2) * sizeof(uchar3)>>>(d_original_image, d_energy_image, width, height);
        
        // Find all seam (from top row to bottom row)
        for(int row = 1; row < height; row++)
            find_seam<<<grid_size1d, block_size1d>>>(d_energy_image, d_back_tracking, width, height, row);

        // Find min seam
        // CHECK(cudaMemcpy(accum_last_row, d_energy_image + width * (height - 1), n_bytes_row,cudaMemcpyDeviceToHost));
        // uint32_t tempp = UINT32_MAX;
        // uint32_t col_start_seam2;

        // for(int i = 0; i < width; i++){
        //     if(accum_last_row[i] < tempp){
        //         tempp = accum_last_row[i];
        //         col_start_seam2 = i;
        //     }
        // }
        find_min_index<<<grid_size1d2, block_size1d, 3 * block_size1d.x * sizeof(uint32_t)>>>(d_energy_image + width * (height - 1), d_minn, width, d_min_indices);
        CHECK(cudaMemcpy(minn, d_minn, grid_size1d2.x * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(min_indices, d_min_indices, grid_size1d2.x * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        for(int i = 0; i < grid_size1d2.x; i++){
            if(minn[i] < minn[0]){
                minn[0] = minn[i];
                min_indices[0] = min_indices[i];
            }
        }
        col_start_seam = min_indices[0];
        // if(col_start_seam!=col_start_seam2){
        //     printf("Col start seam host: %u\n", col_start_seam2);
        //     printf("Col start seam device: %u\n", col_start_seam);
        // }
        // Get seam to delete from backtracking
        CHECK(cudaMemcpy(back_tracking, d_back_tracking, n_bytes_uint32t, cudaMemcpyDeviceToHost));
        seam[height - 1] = col_start_seam;
        for(int row = height - 1; row > 0; row--)
            seam[row - 1] = back_tracking[row * width + seam[row]];

        // Remove seam from image
        CHECK(cudaMemcpy(d_seam, seam, n_bytes_height, cudaMemcpyHostToDevice));
        remove_seam<<<grid_size2d, block_size2d>>>(d_original_image, d_output_image, width, height, d_seam);
        uchar3* temp = d_original_image;
        d_original_image = d_output_image;
        d_output_image = temp;
        width -= 1;
    }
    
    CHECK(cudaMemcpy(out_image, d_original_image, sizeof(uchar3) * width * height, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_original_image));
    CHECK(cudaFree(d_energy_image));
    CHECK(cudaFree(d_back_tracking));
    CHECK(cudaFree(d_seam));
    CHECK(cudaFree(d_output_image));
    CHECK(cudaFree(d_minn));
    CHECK(cudaFree(d_min_indices));
    free(minn);
    free(min_indices);
    free(seam);
    free(back_tracking);
    free(accum_last_row);
}

int main(int argc, char** argv){
    printDeviceInfo();
	int width, height;
	uchar3* original_image;
	readPnm(argv[1], width, height, original_image);
	char* file_name_out  = strtok(argv[1], ".");
	// int n_seams = 5;
	int n_seams = argc == 3 ? atoi(argv[2]) : 100;
	printf("Image size (width x height): %i x %i\n\n", width, height);
    uchar3 *output_image = (uchar3*)malloc((width - n_seams) * height * sizeof(uchar3));

    GpuTimer timer;
    timer.Start();
    remove_n_seam(original_image, output_image, width, height, n_seams);
	timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
	printf("Output image size (width x height): %i x %i\n\n", width - n_seams, height);

    writePnm(output_image, width - n_seams, height, concatStr(file_name_out,"_device5.pnm"));

    free(original_image);
    free(output_image);
    return EXIT_SUCCESS;
}