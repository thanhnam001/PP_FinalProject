#include "support.h"

__global__ void convert_RGB_to_gray(uchar3* original_image, uint8_t* gray_image, int width, int height){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if(row < height && col < width){
        int i = row * width + col;
        gray_image[i] = 0.299f * original_image[i].x + 0.587f * original_image[i].y + 0.114f * original_image[i].z;
    }
}

__global__ void conv_sobel(uint8_t* gray_image, uint32_t* energy_image, int width, int height){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    short int x_sobel[9] = { 1,  0, -1,
					         2,  0, -2,
					         1,  0, -1};
	short int y_sobel[9] = { 1,  2,  1,
					         0,  0,  0,
					        -1, -2, -1};
    if(row < height && col < width){
        int x = 0, y = 0;
        for(int r = 0; r < 3; r++){
            for(int c = 0; c < 3; c++){
                int rc = min(max(row - 1 + r, 0), height - 1); 
                int cc = min(max(col - 1 + c, 0), width - 1);
                x += gray_image[rc * width + cc] * x_sobel[r * 3 + c];
                y += gray_image[rc * width + cc] * y_sobel[r * 3 + c];
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
__device__ int b_count = 0;
volatile __device__ int b_count1 = 0;
__global__ void find_min_indices(uint32_t* last_row, uint32_t* minn, int n, volatile uint32_t* min_indices){
    extern __shared__ uint32_t s_data[];
    __shared__ int bi;
    if(threadIdx.x == 0)
        bi = atomicAdd(&b_count, 1);
    __syncthreads();
    int i = bi * blockDim.x + threadIdx.x;
    if(i < n){
        s_data[threadIdx.x] = last_row[i];
        min_indices[i] = i; 
    }
    else{
        s_data[threadIdx.x] = UINT32_MAX;
        min_indices[i] = 0; 
    }
    __syncthreads();
    for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1){
        int needed_val, needed_idx;
        if(threadIdx.x <= stride && threadIdx.x +stride < blockDim.x){
            needed_val = s_data[threadIdx.x + stride];
            needed_idx = min_indices[i+stride];
        }
        __syncthreads();
        if(threadIdx.x <= stride && threadIdx.x +stride < blockDim.x){
            if(s_data[threadIdx.x] > needed_val){
                s_data[threadIdx.x] = needed_val;
                min_indices[i] = needed_idx;
            } else if(s_data[threadIdx.x] == needed_val)
                min_indices[i] = min(needed_idx, min_indices[i]);
        }
        __syncthreads();
    }
    if(threadIdx.x == 0)
        minn[bi] = s_data[0];
    __syncthreads();
    if(threadIdx.x == 0){
        if(bi > 0){
            while(b_count1 < bi){}
            if(minn[0] > minn[bi]){
                minn[0] = minn[bi];
                min_indices[0] = min_indices[bi * blockDim.x];
            }
            __threadfence();
        }
        b_count1 += 1;
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
    uchar3 *d_original_image;
    uint8_t *d_gray_image;
    uint32_t *d_energy_image;
    uint32_t *d_back_tracking;
    uint32_t *d_seam;
    uchar3 * d_output_image;
    uint32_t *minn, *min_indices;
    size_t n_bytes_uchar3 = width * height * sizeof(uchar3);
    size_t n_bytes_uint8t = width * height * sizeof(uint8_t);
    size_t n_bytes_uint32t = width * height * sizeof(uint32_t);
    size_t n_bytes_row = width * sizeof(uint32_t);
    size_t n_bytes_height = height * sizeof(uint32_t);
    CHECK(cudaMalloc(&d_original_image, n_bytes_uchar3));
    CHECK(cudaMalloc(&d_gray_image, n_bytes_uint8t));
    CHECK(cudaMalloc(&d_energy_image, n_bytes_uint32t));
    CHECK(cudaMalloc(&d_back_tracking, n_bytes_uint32t));
    CHECK(cudaMalloc(&d_seam, n_bytes_height));
    CHECK(cudaMalloc(&d_output_image, n_bytes_uchar3));
    CHECK(cudaMemcpy(d_original_image, original_image, n_bytes_uchar3, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&minn, n_bytes_row));
    CHECK(cudaMalloc(&min_indices, n_bytes_row));
    uint32_t* seam = (uint32_t*)malloc(n_bytes_height);
    uint32_t* back_tracking = (uint32_t*)malloc(n_bytes_uint32t);
    dim3 block_size2d(32,32);
    dim3 grid_size2d((width - 1) / block_size2d.x + 1, (height - 1) / block_size2d.y + 1);
    dim3 block_size1d(512);
    dim3 grid_size1d((width - 1) / block_size1d.x + 1);
    const int z = 0;
    uint32_t col_start_seam;
    for(int i = 0; i < n_seams; i++){
        CHECK(cudaMemcpyToSymbol(b_count , &z, sizeof(int)));
        CHECK(cudaMemcpyToSymbol(b_count1, &z, sizeof(int)));

        // Convert RGB to gray
        convert_RGB_to_gray<<<grid_size2d, block_size2d>>>(d_original_image, d_gray_image, width, height);

        // Calculate energy image
        conv_sobel<<<grid_size2d, block_size2d>>>(d_gray_image, d_energy_image, width, height);
        
        // Find all seam (from top row to bottom row)
        for(int row = 1; row < height; row++)
            find_seam<<<grid_size1d, block_size1d>>>(d_energy_image, d_back_tracking, width, height, row);

        // Find min seam
        find_min_indices<<<grid_size1d, block_size1d, block_size1d.x * sizeof(uint32_t)>>>(d_energy_image + width * (height - 1), minn, width, min_indices);
        CHECK(cudaMemcpy(&col_start_seam, min_indices, sizeof(uint32_t), cudaMemcpyDeviceToHost));

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
    CHECK(cudaFree(d_gray_image));
    CHECK(cudaFree(d_energy_image));
    CHECK(cudaFree(d_back_tracking));
    CHECK(cudaFree(d_seam));
    CHECK(cudaFree(d_output_image));
    CHECK(cudaFree(minn));
    CHECK(cudaFree(min_indices));
    free(seam);
    free(back_tracking);
}

int main(int argc, char** argv){
    printDeviceInfo();
	int width, height;
	uchar3* original_image;
	readPnm(argv[1], width, height, original_image);
	char* file_name_out  = strtok(argv[1], ".");
	int n_seams = argc == 3 ? atoi(argv[2]) : 100;
	printf("Image size (width x height): %i x %i\n\n", width, height);
    uchar3 *output_image = (uchar3*)malloc((width - n_seams) * height * sizeof(uchar3));

    GpuTimer timer;
    timer.Start();
    remove_n_seam(original_image, output_image, width, height, n_seams);
	timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());

    writePnm(output_image, width - n_seams, height, concatStr(file_name_out,"_device1.pnm"));

    free(original_image);
    free(output_image);
    return EXIT_SUCCESS;
}