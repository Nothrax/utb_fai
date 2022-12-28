/**
 * Simple CUDA application template.
 */
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <png++/png.hpp>

#define BLOCK_SIZE (16)
#define FILTER_ROW (3)
#define TILE_SIZE (12)

float filterBlur[FILTER_ROW][FILTER_ROW] = {
		{ 0.111, 0.111, 0.111 },
		{ 0.111, 0.111, 0.111 },
		{ 0.111, 0.111, 0.111 }
};

float filterCopy[FILTER_ROW][FILTER_ROW] = {
		{ 0, -1, 0 },
		{ -1, 4, -1 },
		{ 0, -1, 0 },
};

float filterEdgeDetect[FILTER_ROW][FILTER_ROW] = {
		{ -1, -2, 1 },
		{ 0, 0, 0 },
		{ -1, -2, 1 },
};


/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {                            \
    cudaError_t err = value;                                    \
    if( err != cudaSuccess ) {                                    \
        fprintf( stderr, "Error %s at line %d in file %s\n",    \
                cudaGetErrorString(err), __LINE__, __FILE__ );    \
        exit( 1 );                                                \
    } }

__global__ void
processImage(float *out, const float *in, unsigned int width, unsigned int height, const float *filter,
			 unsigned int filterSize) {

	int x_o = TILE_SIZE*blockIdx.x + threadIdx.x;
	int y_o = TILE_SIZE*blockIdx.y + threadIdx.y;

	int x_i = x_o - 2;
	int y_i = y_o - 2;
	__shared__ float sharedBuffer[BLOCK_SIZE][BLOCK_SIZE];

	if((x_i >= 0) && (x_i < width) && (y_i >= 0) && (y_i < height)) {
		sharedBuffer[threadIdx.y][threadIdx.x] = in[y_i*width + x_i];
	} else {
		sharedBuffer[threadIdx.y][threadIdx.x] = 0;
	}

	__syncthreads();
	float sum = 0;

	if(threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE) {
		for(int row = 0; row < filterSize; ++row) {
			for(int column = 0; column < filterSize; ++column) {
				sum += sharedBuffer[threadIdx.y + row][threadIdx.x + column]*filter[filterSize*row + column];
			}
		}
		if((x_o < width) && (y_o < height)) {
			out[y_o*width + x_o] = sum;
		}
	}
}

struct ImageBuffer {
	float *r { nullptr };
	float *g { nullptr };
	float *b { nullptr };
	ulong size { 0 };

	void allocHost(const ulong sizeToAlloc) {
		size = sizeToAlloc;
		r = new float[size];
		g = new float[size];
		b = new float[size];
	};

	void allocDevice(const ulong sizeToAlloc) {
		size = sizeToAlloc;
		CUDA_CHECK_RETURN(cudaMalloc(&r, size));
		CUDA_CHECK_RETURN(cudaMalloc(&g, size));
		CUDA_CHECK_RETURN(cudaMalloc(&b, size));
	};

	void deallocHost() {
		delete[] r;
		delete[] g;
		delete[] b;
		size = 0;
	}

	void deallocDevice() {
		CUDA_CHECK_RETURN(cudaFree(r));
		CUDA_CHECK_RETURN(cudaFree(g));
		CUDA_CHECK_RETURN(cudaFree(b));
		size = 0;
	}

};

void pngToRgb3(ImageBuffer imageBuffer, const png::image<png::rgb_pixel> &imgPng);

void rgb3ToPng(png::image<png::rgb_pixel> &imgPng, ImageBuffer imageBuffer);
void rgb3ToGrayscale(ImageBuffer imageBuffer);

int main(int argc, char **argv) {
	try{
	std::string filepath;
	int mode;

	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		if (arg == "-h") {
			std::cout << "Usage: proj3 [-h] [-f filename] [-m 1-3]" << std::endl;
			return 0;
		} else if (arg == "-f") {
			if (i + 1 < argc) {
				filepath = argv[i + 1];
				i++;
			} else {
				std::cerr << "-f option requires a filename" << std::endl;
			}
		} else if (arg == "-m") {
			if (i + 1 < argc) {
				// Make sure there is a following argument
				try{
					mode = std::stoi(argv[i + 1]);
				}catch(...){
					mode = -1;
				}
				if(mode < 1 || mode > 3){
					std::cerr << "mode can be only 1 for blur, 2 and 3 for edge detection\n";
					return 1;
				}
				i++;
			} else {
				std::cerr << "-m option requires a number 1-3" << std::endl;
			}
		} else {
			std::cerr << "Unknown argument: " << arg << std::endl;
			return 1;
		}
	}


	png::image<png::rgb_pixel> image(filepath);
	ImageBuffer hostSource, hostTarget;

	ImageBuffer deviceSource, deviceTarget;
	float *deviceFilter { nullptr };

	auto imageSize = image.get_width()*image.get_height()*sizeof(float);
	auto filterSize = FILTER_ROW*FILTER_ROW*sizeof(float);
	hostSource.allocHost(imageSize);
	hostTarget.allocHost(imageSize);

	pngToRgb3(hostSource, image);

	deviceSource.allocDevice(imageSize);
	deviceTarget.allocDevice(imageSize);

	if(mode != 1){
		rgb3ToGrayscale(hostSource);
	}


	CUDA_CHECK_RETURN(cudaMalloc(&deviceFilter, filterSize));
	CUDA_CHECK_RETURN(cudaMemcpy(deviceSource.r, hostSource.r, deviceSource.size, cudaMemcpyHostToDevice))
	CUDA_CHECK_RETURN(cudaMemcpy(deviceSource.g, hostSource.g, deviceSource.size, cudaMemcpyHostToDevice))
	CUDA_CHECK_RETURN(cudaMemcpy(deviceSource.b, hostSource.b, deviceSource.size, cudaMemcpyHostToDevice))

	switch(mode) {
		case 1:
		CUDA_CHECK_RETURN(cudaMemcpy(deviceFilter, filterBlur, filterSize, cudaMemcpyHostToDevice))
			break;
		case 2:
		CUDA_CHECK_RETURN(cudaMemcpy(deviceFilter, filterCopy, filterSize, cudaMemcpyHostToDevice))
			break;
		case 3:
		CUDA_CHECK_RETURN(cudaMemcpy(deviceFilter, filterEdgeDetect, filterSize, cudaMemcpyHostToDevice))
			break;
		default:
		CUDA_CHECK_RETURN(cudaMemcpy(deviceFilter, filterCopy, filterSize, cudaMemcpyHostToDevice))
			break;
	}

	dim3 grid_size((image.get_width() + TILE_SIZE - 1)/TILE_SIZE, (image.get_height() + TILE_SIZE - 1)/TILE_SIZE);
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);

	processImage<<< grid_size, block_size >>>(deviceTarget.r, deviceSource.r, image.get_width(), image.get_height(),
											  deviceFilter, FILTER_ROW);
	processImage<<< grid_size, block_size >>>(deviceTarget.g, deviceSource.g, image.get_width(), image.get_height(),
											  deviceFilter, FILTER_ROW);
	processImage<<< grid_size, block_size >>>(deviceTarget.b, deviceSource.b, image.get_width(), image.get_height(),
											  deviceFilter, FILTER_ROW);


	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	CUDA_CHECK_RETURN(cudaMemcpy(hostTarget.r, deviceTarget.r, hostTarget.size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(hostTarget.g, deviceTarget.g, hostTarget.size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(hostTarget.b, deviceTarget.b, hostTarget.size, cudaMemcpyDeviceToHost));

	rgb3ToPng(image, hostTarget);

	image.write("./processed_image.png");

	hostSource.deallocHost();
	hostTarget.deallocHost();

	deviceSource.deallocDevice();
	deviceTarget.deallocDevice();
	CUDA_CHECK_RETURN(cudaFree(deviceFilter));

	}catch(...){
		std::cout << "Usage: proj3 [-h] [-f filename] [-m 1-3]" << std::endl;
		return 1;
	}
	return 0;
}

void rgb3ToPng(png::image<png::rgb_pixel> &imgPng, ImageBuffer imageBuffer) {
	unsigned int width = imgPng.get_width();
	unsigned int height = imgPng.get_height();

	for(unsigned int y = 0; y < height; ++y)
		for(unsigned int x = 0; x < width; ++x) {
			imgPng.set_pixel(x, y, png::rgb_pixel(*imageBuffer.r++,
												  *imageBuffer.g++,
												  *imageBuffer.b++));
		}
}

void pngToRgb3(ImageBuffer imageBuffer, const png::image<png::rgb_pixel> &imgPng) {
	unsigned int height = imgPng.get_height();

	for(unsigned int y = 0; y < height; ++y) {
		std::vector<png::rgb_pixel> row = imgPng.get_row(y);
		for(std::vector<png::rgb_pixel>::iterator it = row.begin(); it != row.end(); ++it) {
			png::rgb_pixel pixel = *it;

			*imageBuffer.r++ = pixel.red;
			*imageBuffer.g++ = pixel.green;
			*imageBuffer.b++ = pixel.blue;
		}
	}
}

void rgb3ToGrayscale(ImageBuffer imageBuffer) {
	float sum;
	for(unsigned int i = 0; i < imageBuffer.size; i++) {
		sum = imageBuffer.r[i] + imageBuffer.g[i] +imageBuffer.b[i];
		imageBuffer.r[i] = sum/3;
		imageBuffer.g[i] = sum/3;
		imageBuffer.b[i] = sum/3;
	}
}