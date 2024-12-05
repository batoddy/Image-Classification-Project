
// model: y = f(w*x)
// x: 256*256
// w : 1

// includes
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h" // reading jpg image library

// defines
#define dim 32
#define len (dim * dim + 1)

#define EPOCHS 500
#define BACHES_SGD 10

#define lr 0.01 // learning rate (n)
#define W 0.01
#define CAR_LBL -1.0
#define TANK_LBL 1.0
#define EPOCH_PRINT 20

#define CAR_TRAIN_START 0
#define TANK_TRAIN_START 0
#define CAR_TEST_START 735
#define TANK_TEST_START 568

#define TRAIN_DATA_NBR 500
#define TEST_DATA_NBR 100

#define SOI 0xFFD8
#define SOS 0xFFDA
#define EOI 0xFFD9

typedef enum TYPE
{
    TRAIN_CAR = 0,
    TRAIN_TANK,
    TEST_CAR,
    TEST_TANK
} TYPE;

// func prototypes
void read_img(int, double *, TYPE);
double dotProduct(double *, double *, int);
double *gradient_descent_optimization(double *, uint64_t *, double[len][TRAIN_DATA_NBR], double[len][TRAIN_DATA_NBR]);
double *sgd_optimization(double *, uint64_t *, double[len][TRAIN_DATA_NBR], double[len][TRAIN_DATA_NBR]);
double adam_optimization(double *, uint64_t *);

void print_vector(double *, int);
void fill_parameters(double *vector, double value, int n);

double model(double *w, double *x, int n);
double test_model(double *w, double x[len][TRAIN_DATA_NBR], int n, double lbl, int data_size);

char train_car_file_path[100] = "GRAY-cars_tanks/resized/train/cars";
char train_tank_file_path[100] = "GRAY-cars_tanks/resized/train/tanks";

char test_car_file_path[100] = "GRAY-cars_tanks/resized/test/cars";
char test_tank_file_path[100] = "GRAY-cars_tanks/resized/test/tanks";

uint64_t getTick()
{
    // printf("TIME:%d\n", time(NULL));
    return time(NULL);
}
// double tanh(double x) { return tanh(x); }

double img_mtrx_car_train[len][TRAIN_DATA_NBR];
double img_mtrx_tank_train[len][TRAIN_DATA_NBR];
double img_mtrx_car_test[len][TEST_DATA_NBR];
double img_mtrx_tank_test[len][TEST_DATA_NBR];

// main code
int main()
{

    // img datas
    double *gd_weights;
    double *sgd_weights;
    double *adam_weights;

    // variables
    double loss_arr_GD[EPOCHS];
    uint64_t time_arr_GD[EPOCHS];
    double loss_arr_SGD[EPOCHS];
    uint64_t time_arr_SGD[EPOCHS];
    double loss_arr_ADAM[EPOCHS];
    uint64_t time_arr_ADAM[EPOCHS];

    for (int i = 0; i < TRAIN_DATA_NBR; i++) // fill car train img matrix,
        read_img(i + CAR_TRAIN_START, &img_mtrx_car_train[i][0], TRAIN_CAR);

    for (int i = 0; i < TRAIN_DATA_NBR; i++) // fill tank train img matrix
        read_img(i + TANK_TRAIN_START, &img_mtrx_tank_train[i][0], TRAIN_TANK);

    for (int i = 0; i < TEST_DATA_NBR; i++) // fill car test img matrix
        read_img(i + CAR_TEST_START, &img_mtrx_car_test[i][0], TEST_CAR);

    for (int i = 0; i < TEST_DATA_NBR; i++) // fill tank test img matrix
        read_img(i + TANK_TEST_START, &img_mtrx_tank_test[i][0], TEST_TANK);
    //--

    printf("read_finished\n");

    gd_weights = gradient_descent_optimization(loss_arr_GD, time_arr_GD, img_mtrx_car_train, img_mtrx_tank_train);

    sgd_weights = sgd_optimization(loss_arr_SGD, time_arr_SGD, img_mtrx_car_train, img_mtrx_tank_train);

    printf("GD_CAR_TEST\n");
    test_model(gd_weights, img_mtrx_car_test, len, CAR_LBL, TEST_DATA_NBR);
    printf("GD_TANK_TEST\n");
    test_model(gd_weights, img_mtrx_tank_test, len, TANK_LBL, TEST_DATA_NBR);
    printf("GD_CAR_TRAIN\n");
    test_model(gd_weights, img_mtrx_car_train, len, CAR_LBL, TRAIN_DATA_NBR);
    printf("GD_TANK_TRAIN\n");
    test_model(gd_weights, img_mtrx_tank_train, len, TANK_LBL, TRAIN_DATA_NBR);

    printf("SGD_CAR_TEST\n");
    test_model(sgd_weights, img_mtrx_car_test, len, CAR_LBL, TEST_DATA_NBR);
    printf("SGD_TANK_TEST\n");
    test_model(sgd_weights, img_mtrx_tank_test, len, TANK_LBL, TEST_DATA_NBR);
    printf("SGD_CAR_TRAIN\n");
    test_model(sgd_weights, img_mtrx_car_train, len, CAR_LBL, TRAIN_DATA_NBR);
    printf("SGD_TANK_TRAIN\n");
    test_model(sgd_weights, img_mtrx_tank_train, len, TANK_LBL, TRAIN_DATA_NBR);

    // for (int i = 0; i < TEST_DATA_NBR; i++)
    // {
    //     printf("Tank: %f\n", model(sgd_weights, img_mtrx_tank_test[i], len));
    //     printf("Car: %f\n", model(sgd_weights, img_mtrx_car_test[i], len));
    // }
    // for (int i = 0; i < EPOCHS; i++)
    // {
    //     printf("Epoch %d || loss: %f time:%d\n", i + 1, loss_arr_GD[i], time_arr_GD[i]);
    // }

    // for (int i = 0; i < EPOCHS; i++)
    // {
    //     printf("Epoch %d || loss: %f time:%d\n", i + 1, loss_arr_SGD[i], time_arr_SGD[i]);
    // }
    // // adam
    return 0;
}

void read_img(int img_no, double *x, TYPE type) // read img func - temporarly loads t to img
{
    FILE *fp;
    char file_path[100];
    int img_len = len - 1;
    uint8_t pixel;
    uint16_t prev_pixel;
    long file_size, payload_size = 0;
    uint8_t img_matrix[len - 1];
    double lbl;

    int width, height, n; // img read lib's variables
    uint8_t *img_data;    // -
    uint8_t r, g, b;
    double grayscale;

    switch (type)
    {
    case TRAIN_CAR:
        sprintf(file_path, "%s/%d.jpg", train_car_file_path, img_no);
        lbl = CAR_LBL;
        break;
    case TRAIN_TANK:
        sprintf(file_path, "%s/%d.jpg", train_tank_file_path, img_no);
        lbl = TANK_LBL;
        break;
    case TEST_CAR:
        sprintf(file_path, "%s/%d.jpg", test_car_file_path, img_no);
        lbl = CAR_LBL;
        break;
    case TEST_TANK:
        sprintf(file_path, "%s/%d.jpg", test_tank_file_path, img_no);
        lbl = TANK_LBL;
        break;
    }

    img_data = (uint8_t *)stbi_load(file_path, &width, &height, &n, 0);
    if (img_data != NULL)
    {
        if (width != dim || height != dim)
            printf("Image dimention error!! (width: %d, height: %d)\n", width, height);

        int index = 0; // Vektöre yazılacak indeks
        for (int row = 0; row < width; row++)
        {
            for (int col = 0; col < height; col++)
            {

                r = img_data[(row * width + col) * 3];     // Kırmızı kanal
                g = img_data[(row * width + col) * 3 + 1]; // Yeşil kanal
                b = img_data[(row * width + col) * 3 + 2]; // Mavi kanal

                grayscale = (r + g + b) / 3.0 / 255.0; // Normalize [0,1] aralığına
                // printf(" index =%d grayscale: %f",index, grayscale);
                x[index] = grayscale;
                // printf(" %d ye %d x= %f\n " , gozlem,index, X[gozlem][index]);

                index++;
            }
        }
        x[index] = lbl; // add bias

        stbi_image_free(img_data);
    }
    else
    {
        printf("Image read error!!\n");
        exit(1);
    }

    // printf("File path: %s\n", file_path);

    // fp = fopen(file_path, "rb");

    // fseek(fp, 0, SEEK_END); // get file size
    // file_size = ftell(fp);
    // // printf("File size: %d bytes \n", file_size);
    // fseek(fp, 0, SEEK_SET);

    // if (file_size > 100000)
    //     printf("Image too big\n");

    // fread(&pixel, sizeof(uint8_t), 1, fp); // read by words
    // prev_pixel = (prev_pixel << 8) | pixel;
    // fread(&pixel, sizeof(uint8_t), 1, fp); // read by words
    // prev_pixel = (prev_pixel << 8) | pixel;

    // if (prev_pixel != SOI)
    //     printf("Image read error!! - %X\n", prev_pixel);

    // // else if (prev_pixel == SOI)
    // //     printf("Image SOF read. \n");

    // // printf("\n==HAEADER==\n");
    // while (prev_pixel != SOS)
    // {
    //     fread(&pixel, sizeof(uint8_t), 1, fp); // read by words
    //     prev_pixel = (prev_pixel << 8) | pixel;
    //     // printf("%.2X ", pixel);
    // }
    // int j = 0;
    // // printf("\n==IMAGE DATA==\n");
    // while (prev_pixel != EOI && j < len)
    // {
    //     payload_size++;
    //     fread(&pixel, sizeof(uint8_t), 1, fp); // read by words
    //     prev_pixel = (prev_pixel << 8) | pixel;
    //     x[j] = (double)pixel / 255.0;
    //     j++;
    //     // printf("%.2X ", pixel);
    //     // printf("x%d: %f\n", j, x[j]);
    // }
    // // printf("Payload: %d\n", payload_size);

    // fclose(fp);
}

void print_vector(double *vector, int n) // printing vector func
{
    printf("[");
    for (int i = 0; i < n; i++)
    {
        printf("%.2f ,", vector[i]);
    }
    printf("]\n");
}

void fill_parameters(double *vector, double value, int n)
{
    for (int i = 0; i < n; i++)
    {
        vector[i] = value;
    }
}

double dotProduct(double *x, double *w, int n) // get dot product func
{
    double res = 0;
    for (int i = 0; i < n; i++)
    {
        res += w[i] * x[i];

        // printf("i:%d w:%f x:%f res:%f\n", i, w[i], x[i], res);
    }
    // printf("dotProduct: %lf\n", res);
    return res;
}

void add_to_fifo(double *fifo, double buffer, int fifo_len)
{
    for (int i = fifo_len; i > 0; i--)
    {
        fifo[i] = fifo[i - 1];
    }
    fifo[0] = buffer;
}

double *gradient_descent_optimization(double *loss_arr, uint64_t *time_arr, double car_matrix[len][TRAIN_DATA_NBR], double tank_matrix[len][TRAIN_DATA_NBR])
{
    static double w[len];
    double y = 0;
    double t;
    double error, loss, gradient, loss_sum;
    uint64_t tick, tick_c;

    double *x = malloc(len * sizeof(double));
    if (x == NULL)
    {
        printf("Memory allocation failed for x\n");
        exit(1);
    }
    // n*n+1
    fill_parameters(w, W, len);

    // for (int i = 0; i < len; i++) // fills random w
    // {
    //     w[i] = ((double)rand() / RAND_MAX) * 0.1 - 0.05; // Small random values between -0.05 and 0.05
    // }

    for (int epoch = 1; epoch <= EPOCHS; epoch++)
    {
        // printf("w[%d]: ", epoch);
        // print_vector(w, len);
        // printf("\n\n\n");
        loss_sum = 0;
        // tick = getTick();
        // printf("Tick1:%d ", getTick());
        for (int i = 0; i < (TRAIN_DATA_NBR * 2); i++)
        {
            if (i % 2 == 0)
            {
                x = &car_matrix[0][(int)(i / 2)];
                t = CAR_LBL;
                // printf("C\n");
            }
            else
            {
                x = &tank_matrix[0][(int)(i / 2) - 1];
                t = TANK_LBL;
                // printf("T\n");
            }
            y = tanh(dotProduct(x, w, len)); // calculate y_i
            // printf("y:%f t:%f\n", y, t);
            error = y - t;
            loss = error * error;
            loss_sum += loss;
            for (int j = 0; j < len; j++) // main iteration porcess
            {
                gradient = 2 * error * (1 - (y * y)) * x[j];
                w[j] = w[j] - lr * gradient;
                // printf("Gradient: %f | %f\n", y, t);
                // printf("j:%d/%d\n", j, len);
            }
            // printf("%d-%d: x[%d]:%f, w[%d]:%f, y: %.2f, err:%f\n\n", epoch, i, i, x[i], i, w[i], y, error);
            // printf("Input: %lf == %lf\n", t, y);
        }
        // tick_c = getTick();

        if (epoch % EPOCH_PRINT == 0 || epoch == 0)
            printf("GD ||\tepoch(%d/%d) | loss:%lf\n", epoch, EPOCHS, loss_sum / (TRAIN_DATA_NBR * 2));

        loss_arr[epoch] = loss_sum / (TRAIN_DATA_NBR * 2);
        time_arr[epoch] = tick_c - tick;
    }
    // printf("tick2: %d", tick_c - tick);
    return w;
}

double *sgd_optimization(double *loss_arr, uint64_t *time_arr, double car_matrix[len][TRAIN_DATA_NBR], double tank_matrix[len][TRAIN_DATA_NBR])
{
    static double w[len];
    double y = 0;
    double t;
    double error, loss, gradient, loss_sum, loss_sum_bached, gradients[len] = {0};
    double error_arr[BACHES_SGD];
    uint64_t tick, tick_c;

    double *x = malloc(len * sizeof(double));
    if (x == NULL)
    {
        printf("Memory allocation failed for x\n");
        exit(1);
    }
    // n*n+1

    // for (int i = 0; i < len; i++) // fills random w
    // {
    //     w[i] = ((double)rand() / RAND_MAX) * 0.1 - 0.05; // Small random values between -0.05 and 0.05
    // }

    fill_parameters(w, W, len);

    for (int epoch = 1; epoch <= EPOCHS; epoch++)
    {
        // printf("w[%d]: ", epoch);
        // print_vector(w, len);
        // printf("\n\n\n");
        loss_sum = 0;
        // tick = getTick();
        // printf("Tick1:%d ", getTick());
        for (int i = 0; i < (TRAIN_DATA_NBR * 2); i++)
        {
            if (i % 2 == 0)
            {
                x = &car_matrix[0][(int)(i / 2)];
                t = CAR_LBL;
                // printf("C\n");
            }
            else
            {
                x = &tank_matrix[0][(int)(i / 2) - 1];
                t = TANK_LBL;
                // printf("T\n");
            }
            y = tanh(dotProduct(x, w, len)); // calculate y_i

            error = y - t;
            loss = error * error;
            loss_sum += loss;

            add_to_fifo(error_arr, loss, BACHES_SGD);

            // printf("y:%f t:%f |||", y, t);

            for (int j = 0; j < len; j++)
            {
                gradients[j] += 2 * error * (1 - (y * y)) * x[j]; // Gradyanları biriktir
            }

            // Eğer bir batch tamamlandıysa ağırlıkları güncelle
            if ((i + 1) % BACHES_SGD == 0 || i == (TRAIN_DATA_NBR * 2) - 1)
            {
                for (int j = 0; j < len; j++)
                {
                    w[j] = w[j] - (lr / BACHES_SGD) * gradients[j]; // Ağırlık güncelle
                    gradients[j] = 0;                               // Gradyanı sıfırla
                }

                loss_sum += loss_sum_bached / BACHES_SGD; // Batch'in ortalama loss'unu hesapla
                loss_sum_bached = 0;                      // Batch için loss toplamını sıfırla
            }
        }
        // tick_c = getTick();
        if (epoch % EPOCH_PRINT == 0 || epoch == 0)
            printf("SGD ||\tepoch(%d/%d) | loss:%lf\n", epoch, EPOCHS, loss_sum / (TRAIN_DATA_NBR * 2));

        loss_arr[epoch] = loss_sum / (TRAIN_DATA_NBR * 2);
        time_arr[epoch] = tick_c - tick;
    }
    return w;
}

double adam_optimization(double *loss_arr, uint64_t *time_arr)
{
    // TODO write adam
}

double model(double *w, double *x, int n)
{
    return tanh(dotProduct(w, x, n));
}

double test_model(double *w, double x[len][TRAIN_DATA_NBR], int n, double lbl, int data_size)
{
    double output, true_avg = 0, false_avg = 0;
    int true_predict = 0, false_predict = 0, zero = 0;

    for (int j = 0; j < data_size; j++)
    {
        switch ((int)lbl)
        {
        case (int)TANK_LBL:

            output = model(w, x[j], n);
            if (output > 0)
            {
                // printf("TANK Predict True - %lf\n", output);
                true_predict++;
                true_avg += output;
            }
            else if (output < 0)
            {
                // printf("TANK Predict False - %lf\n", output);
                false_predict++;
                false_avg++;
            }
            else
            {
                // printf("TANK can't predict - %lf j:%d\n", output, j);
                // print_vector(x[j], n);
                zero++;
            }

            break;

        case (int)CAR_LBL:

            output = model(w, x[j], n);
            if (output < 0)
            {
                // printf("CAR Predict True - %lf\n", output);
                true_predict++;
                true_avg += output;
            }
            else if (output > 0)
            {
                // printf("CAR Predict False - %lf\n", output);

                false_predict++;
                false_avg++;
            }
            else
            {
                // printf("CAR can't predict - %lf j:%d\n", output, j);
                // print_vector(x[j], n);
                zero++;
            }

            break;
        }
    }
    true_avg /= true_predict;
    false_avg /= false_predict;

    printf("Predict (%d/%d)\t| True avg: %.3lf False_avg: %.3lf zero: %d\n", true_predict, data_size, true_avg, false_avg, zero);
}
