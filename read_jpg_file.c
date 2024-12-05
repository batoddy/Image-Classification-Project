#include <stdio.h>
#include <stdint.h>
#define FILE_PATH "114921_2.jpg"
#define N 50
#define SOI 0xFFD8
#define SOS 0xFFDA
#define EOI 0xFFD9

uint16_t read_word(FILE *fptr);

int main()
{

    FILE *fp;
    long file_size, payload_size;
    int index;
    uint8_t pixel;
    uint16_t prev_pixel;

    fp = fopen(FILE_PATH, "rb");

    fseek(fp, 0, SEEK_END); // get file size
    file_size = ftell(fp);
    printf("File size: %d bytes \n", file_size);
    fseek(fp, 0, SEEK_SET);

    fread(&pixel, sizeof(uint8_t), 1, fp); // read by words
    prev_pixel = (prev_pixel << 8) | pixel;
    fread(&pixel, sizeof(uint8_t), 1, fp); // read by words
    prev_pixel = (prev_pixel << 8) | pixel;

    if (prev_pixel != SOI)
        printf("Image read error!! - %X\n", prev_pixel);

    else if (prev_pixel == SOI)
        printf("Image SOF read. \n");

    printf("\n==HAEADER==\n");
    while (prev_pixel != SOS)
    {
        fread(&pixel, sizeof(uint8_t), 1, fp); // read by words
        prev_pixel = (prev_pixel << 8) | pixel;
        printf("%.2X ", pixel);
    }

    printf("\n==IMAGE DATA==\n");
    while (prev_pixel != EOI)
    {
        payload_size++;
        fread(&pixel, sizeof(uint8_t), 1, fp); // read by words
        prev_pixel = (prev_pixel << 8) | pixel;
        printf("%.2X ", pixel);
    }
    printf("Payload: %d\n", payload_size);

    fclose(fp);

    return 0;
}
