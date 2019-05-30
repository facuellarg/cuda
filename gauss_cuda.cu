
#include <stdlib.h>
#include <stdio.h>

#include "png.h"



png_byte color_type;



int main(int argc, char *argv[])
{
    if (argc != 5)
        abort();
    int tamaño = atoi(argv[3]);
    int radio = floor(tamaño / 2);
    double sigma = radio * radio;
    read_png_file(argv[1]);
    struct timeval start_time, stop_time, elapsed_time;
    gettimeofday(&start_time, NULL);
    int cantidad_hilos = atoi(argv[4]);
    int heightHilos = height;
    omp_set_num_threads(cantidad_hilos);
    int i, j;

    double **kernel;
    kernel = createKernel(tamaño);
#pragma omp parallel for private(i, j)
    for (i = 0; i < height; i++)
    {
        png_bytep row = row_pointers[i];
        for (j = 0; j < width; j++)
        {
            png_bytep px = &(row[j * 4]);
            double redTemp = 0;
            double greenTemp = 0;
            double blueTemp = 0;
            double acum = 0;
            for (int fila = i - radio; fila < i + radio + (tamaño%2); fila++)
            {
                int y = fila < 0 ? 0 : fila < heightHilos ? fila : heightHilos - 1;
                png_bytep rowKernel = row_pointers[y];
                for (int columna = j - radio; columna < j + radio + (tamaño % 2); columna++)
                {
                    int x = columna < 0 ? 0 : columna < width ? columna : width - 1;
                    double weight = kernel[y - i + radio][x - j + radio];
                    png_bytep pxt = &(rowKernel[x * 4]);
                    redTemp += pxt[0] * weight;
                    greenTemp += pxt[1] * weight;
                    blueTemp += pxt[2] * weight;
                    acum += weight;
                }
            }

            px[0] = round(redTemp / acum);
            px[1] = round(greenTemp / acum);
            px[2] = round(blueTemp / acum);
        }
    }
    for (int i = 0; i < tamaño; i++)
        free(kernel[i]);
    free(kernel);
    gettimeofday(&stop_time, NULL);
    timersub(&stop_time, &start_time, &elapsed_time);
    char tiempo[10];
    sprintf(tiempo, "%f", elapsed_time.tv_sec + elapsed_time.tv_usec / 1000000.0);
    write_png_file(argv[2]);
    char text_otuput[100];
    sprintf(text_otuput, "fopenMP\tHilos : %d\t Tamaño del Kernel %s\t Tamaño de la imagen %dpx\t Tiempo %s", cantidad_hilos, argv[3], width, tiempo);
    write_output(text_otuput);

    return 0;
}