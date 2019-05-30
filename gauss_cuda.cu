/*
 * A simple libpng example program
 * http://zarb.org/~gc/html/libpng.html
 *
 * Modified by Yoshimasa Niwa to make it much simpler
 * and support all defined color_type.
 *
 * To build, use the next instruction on OS X.
 * $ brew install libpng
 * $ clang -lz -lpng15 libpng_test.c
 *
 * Copyright 2002-2010 Guillaume Cottenceau.
 *
 * This software may be freely redistributed under the terms
 * of the X11 license.
 * /usr/local/opt/llvm/bin/clang -lz -lpng16  gauss_omp.c -fopenmp -o gauss-omp
 */
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include "math.h"
#include "png.h"
#include <sys/time.h>

int width, height;
char *d_R, *d_G, *d_B;
char *h_R, *h_G, *h_B;
png_byte color_type;
png_byte bit_depth;
png_bytep *row_pointers;
size_t size;
 __global__ void
blurEffect(double **kernel, int height, int width,  char *r,  char *g,char *b, char radius, int size, int operationPerThread)
{
    int index = ((blockDim.x * blockIdx.x + threadIdx.x));
    if( index < size )
    {
        for(int count = 0; count < operationPerThread; count ++){
            int i = (index + count) / width;// fila
            int j = (index + count) % width;//columna
            double redTemp = 0;
            double greenTemp = 0;
            double blueTemp = 0;
            double acum = 0;
            for (int row = i - radius * width; row < i + radius * width + (sizeof(kernel)%2); row = row + radius*width )
            {
                int y = row < 0 ? 0 : row < height ? row : height - 1;
                for (int column = j - radius; column < j + radius + (sizeof(kernel) % 2); column++)
                {
                    int x = column < 0 ? 0 : column < width ? column : width - 1;
                    redTemp += r[y*width + x] * kernel[y - i + radius][x - j + radius];
                    greenTemp += g[y*width + x] * kernel[y - i + radius][x - j + radius];
                    blueTemp += b[y*width + x] * kernel[y - i + radius][x - j + radius];
                    acum += kernel[y - i + radius][x - j + radius];
                }
            }
            r[i*width + j] = round(redTemp / acum);
            g[i*width + j] = round(greenTemp / acum);
            b[i*width + j] = round(blueTemp / acum);
        }
    }
}

void read_png_file(char *filename)
{

    FILE *fp = fopen(filename, "rb");

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png)
        abort();

    png_infop info = png_create_info_struct(png);
    if (!info)
        abort();

    if (setjmp(png_jmpbuf(png)))
        abort();

    png_init_io(png, fp);

    png_read_info(png, info);

    width = png_get_image_width(png, info);
    height = png_get_image_height(png, info);
    color_type = png_get_color_type(png, info);
    bit_depth = png_get_bit_depth(png, info);

    // Read any color_type into 8bit depth, RGBA format.
    // See http://www.libpng.org/pub/png/libpng-manual.txt

    if (bit_depth == 16)
        png_set_strip_16(png);

    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

    // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    if (png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    // These color_type don't have an alpha channel then fill it with 0xff.
    if (color_type == PNG_COLOR_TYPE_RGB ||
        color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

    if (color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++)
    {
        row_pointers[y] = (png_byte *)malloc(png_get_rowbytes(png, info));
    }

    png_read_image(png, row_pointers);

    fclose(fp);
}

void write_png_file(char *filename)
{
    

    FILE *fp = fopen(filename, "wb");
    if (!fp)
        abort();

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png)
        abort();

    png_infop info = png_create_info_struct(png);
    if (!info)
        abort();

    if (setjmp(png_jmpbuf(png)))
        abort();

    png_init_io(png, fp);

    // Output is 8bit depth, RGBA format.
    png_set_IHDR(
        png,
        info,
        width, height,
        8,
        PNG_COLOR_TYPE_RGBA,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    // To remove the alpha channel for PNG_COLOR_TYPE_RGB format,
    // Use png_set_filler().
    //png_set_filler(png, 0, PNG_FILLER_AFTER);

    png_write_image(png, row_pointers);
    png_write_end(png, NULL);

    for (int y = 0; y < height; y++)
    {
        free(row_pointers[y]);
    }
    free(row_pointers);

    fclose(fp);
}
void write_output(char *text)
{
    FILE *file;

    file = fopen("gauss_blur.txt", "a");
    if (file == NULL)
    {
        /* File not created hence exit */
        printf("Unable to create file.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(file, "%s\n", text);
    fclose(file);
}

double **createKernel(int tamanio)
{
    double **matriz = (double **)malloc(tamanio * sizeof(double *));
    for (int i = 0; i < tamanio; i++)
        matriz[i] = (double *)malloc(tamanio * sizeof(double));
    int radio = floor(tamanio / 2);
    double sigma = radio * radio;
    for (int fila = 0; fila < tamanio; fila++)
    {
        for (int columna = 0; columna < tamanio; columna++)
        {
            double square = (columna - radio) * (columna - radio) + (fila - radio) * (fila - radio);
            double weight = (exp(-square / (2 * sigma))) / (3.14159264 * 2 * sigma);
            matriz[fila][columna] = weight;
        }
    }
    return matriz;
}

void getChannels()
{
    for (int i = 0; i < height; i++)
    {
        png_bytep row = row_pointers[i];
        for (int j = 0; j < width; j++)
        {
            png_bytep px = &(row[j * 4]);
            h_R[i * width + j] = px[0];
            h_G[i * width + j] = px[1];
            h_B[i * width + j] = px[2];
        }
    }
}

void makeRowPointer()
{
    for (int i = 0; i < height; i++)
    {
        png_bytep row = row_pointers[i];
        for (int j = 0; j < width; j++)
        {
            png_bytep px = &(row[j * 4]);
            px[0] = h_R[i * width + j];
            px[1] = h_G[i * width + j];
            px[2] = h_B[i * width + j];
        }
    }
}
int main(int argc, char *argv[])
{

    if (argc != 4)
        abort();  
    cudaError_t err = cudaSuccess;
// declarar  la cantidad de hilos segun la gpu
//-------------------------------------------------
    int dev = 0;
    size_t size = sizeof(float);
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int threadsPerBlock = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
	threadsPerBlock = threadsPerBlock*2;
    int blocksPerGrid =   deviceProp.multiProcessorCount;
//-------------------------------------------------
    int tamanio = atoi(argv[2]);
    char radio = floor(tamanio / 2);
    read_png_file(argv[1]);
    struct timeval start_time, stop_time, elapsed_time;
    gettimeofday(&start_time, NULL);
    size = height * width*sizeof(char);
    // Asignar memoria para cpu
    h_R = (char *)malloc(height * width * sizeof(char));
    h_B = (char *)malloc(height * width * sizeof(char));
    h_G = (char *)malloc(height * width * sizeof(char));
    if (h_R == NULL || h_B == NULL || h_G == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    getChannels();
    double **kernel;
    kernel = createKernel(tamanio);
    
    //Asignacion de memoria para cuda
    
    err = cudaMalloc((void **)&d_R, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector R (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_G, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector R (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector R (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Copiar memoria de host a device
    err = cudaMemcpy(d_R, h_R, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector R from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_G, h_G, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector G from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("max threads per block%d\n",deviceProp.maxThreadsPerMultiProcessor);
    printf("launched  threads per block%d\n",( threadsPerBlock));

    int opt = (int)ceil(height * width/ (threadsPerBlock*blocksPerGrid));
    blurEffect<<<blocksPerGrid,threadsPerBlock) >>>(kernel, height, width, d_R, d_G, d_B, radio, height*width, opt);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    for (int i = 0; i < tamanio; i++)
        free(kernel[i]);
    free(kernel);
    makeRowPointer();
    gettimeofday(&stop_time, NULL);
    timersub(&stop_time, &start_time, &elapsed_time);
    char tiempo[10];
    sprintf(tiempo, "%f", elapsed_time.tv_sec + elapsed_time.tv_usec / 1000000.0);
    write_png_file(argv[2]);
    char text_otuput[100];
    sprintf(text_otuput, "fopenMP\tHilos : %d\t Tamaño del Kernel %s\t Tamaño de la imagen %dpx\t Tiempo %s", threadsPerBlock, argv[3], width, tiempo);
    write_output(text_otuput);

    return 0;
}