
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

// #define INFO
#define TEST_INFO
// defines
#define dim 64              // image dimension
#define len (dim * dim + 1) // n*n+1

#define EPOCHS 100
#define EPOCH_PRINT 10

#define lr 0.0001 // learning rate (n)

// different weight parameters
#define W1 -0.01
#define W2 -0.001
#define W3 0.0001
#define W4 0.001
#define W5 0.01
#define W6 -0.01
// SGD
#define BACHES_SGD 5
// ADAM
#define B1 0.9
#define B2 0.999
#define epsilon 1e-8

#define CAR_LBL -1.0 // labels for categories
#define TANK_LBL 1.0

#define CAR_TRAIN_START 0 // image path definers
#define TANK_TRAIN_START 0
#define CAR_TEST_START 735
#define TANK_TEST_START 568

#define TRAIN_DATA_NBR 500 // train data
#define TEST_DATA_NBR 100  // test data

typedef enum TYPE
{
    TRAIN_CAR = 0,
    TRAIN_TANK,
    TEST_CAR,
    TEST_TANK
} TYPE;

// func prototypes
void allocate_matris(double ***, int, int); // allocate memory for image matrix
void free_matris(double **, int, int);      // free allocated memory

void read_img(int, double *, TYPE);                                                        // read image data and fill to [n*n+1][DATA_NBR] two dimension image matrix
double dotProduct(double *, double *, int);                                                // calculate dot product
double *gradient_descent_optimization(double *, uint64_t *, double **, double **, double); // train model with gradient descent
double *sgd_optimization(double *, uint64_t *, double **, double **, double);              // train model with scoastic gradient descent
double *adam_optimization(double *, uint64_t *, double **, double **, double);             // train model with adam

void print_vector(double *, int);            // print vector to terminal
void fill_parameters(double *, double, int); // fill up the matrix with value (used for weight vector)

double model(double *, double *, int);                    // trained model func
double test_model(double *, double **, int, double, int); // test the model

// write to file functions
void write_train_data_to_file(char *, double[EPOCHS], double[EPOCHS], double[EPOCHS], uint64_t[EPOCHS], uint64_t[EPOCHS], uint64_t[EPOCHS], int);
void write_weigths_to_file(double, double *, double *, double *, int);

char train_car_file_path[100] = "GRAY-cars_tanks/resized/train/cars";
char train_tank_file_path[100] = "GRAY-cars_tanks/resized/train/tanks";

char test_car_file_path[100] = "GRAY-cars_tanks/resized/test/cars";
char test_tank_file_path[100] = "GRAY-cars_tanks/resized/test/tanks";

char train_data_file_path[100] = "data/train_data.csv";

uint64_t getTick() { return clock(); }

// main code
int main()
{
    // img datas pointers
    double **img_mtrx_car_train;
    double **img_mtrx_tank_train;
    double **img_mtrx_car_test;
    double **img_mtrx_tank_test;

    // weight vectors
    double *gd_weights;
    double *sgd_weights;
    double *adam_weights;

    double init_weight_arr[6] = {W1, W2, W3, W4, W5, W6};

    // variables
    double loss_arr_GD[EPOCHS];
    uint64_t time_arr_GD[EPOCHS];
    double loss_arr_SGD[EPOCHS];
    uint64_t time_arr_SGD[EPOCHS];
    double loss_arr_ADAM[EPOCHS];
    uint64_t time_arr_ADAM[EPOCHS];

    allocate_matris(&img_mtrx_car_train, len, TRAIN_DATA_NBR); // allocate memory for data matrix
    allocate_matris(&img_mtrx_tank_train, len, TRAIN_DATA_NBR);
    allocate_matris(&img_mtrx_car_test, len, TEST_DATA_NBR);
    allocate_matris(&img_mtrx_tank_test, len, TEST_DATA_NBR);

    for (int i = 0; i < TRAIN_DATA_NBR; i++) // fill car train img matrix,
        read_img(i + CAR_TRAIN_START, img_mtrx_car_train[i], TRAIN_CAR);

    for (int i = 0; i < TRAIN_DATA_NBR; i++) // fill tank train img matrix
        read_img(i + TANK_TRAIN_START, img_mtrx_tank_train[i], TRAIN_TANK);

    for (int i = 0; i < TEST_DATA_NBR; i++) // fill car test img matrix
        read_img(i + CAR_TEST_START, img_mtrx_car_test[i], TEST_CAR);

    for (int i = 0; i < TEST_DATA_NBR; i++) // fill tank test img matrix
        read_img(i + TANK_TEST_START, img_mtrx_tank_test[i], TEST_TANK);
    //--

    printf("Image read finished\n");

    for (int i = 0; i < 6; i++) // train model with 5 different weight initilation value
    {

        gd_weights = gradient_descent_optimization(loss_arr_GD, time_arr_GD, img_mtrx_car_train, img_mtrx_tank_train, init_weight_arr[i]);
        sgd_weights = sgd_optimization(loss_arr_SGD, time_arr_SGD, img_mtrx_car_train, img_mtrx_tank_train, init_weight_arr[i]);
        adam_weights = adam_optimization(loss_arr_ADAM, time_arr_ADAM, img_mtrx_car_train, img_mtrx_tank_train, init_weight_arr[i]);

        write_weigths_to_file(init_weight_arr[i], gd_weights, sgd_weights, adam_weights, len); // write weights to file

        printf("\nGD_CAR_TEST\t");
        test_model(gd_weights, img_mtrx_car_test, len, CAR_LBL, TEST_DATA_NBR);
        printf("GD_CAR_TEST\t");
        test_model(gd_weights, img_mtrx_tank_test, len, TANK_LBL, TEST_DATA_NBR);
        printf("GD_CAR_TRAIN\t");
        test_model(gd_weights, img_mtrx_car_train, len, CAR_LBL, TRAIN_DATA_NBR);
        printf("GD_TANK_TRAIN\t");
        test_model(gd_weights, img_mtrx_tank_train, len, TANK_LBL, TRAIN_DATA_NBR);

        printf("\nSGD_CAR_TEST\t");
        test_model(sgd_weights, img_mtrx_car_test, len, CAR_LBL, TEST_DATA_NBR);
        printf("SGD_TANK_TEST\t");
        test_model(sgd_weights, img_mtrx_tank_test, len, TANK_LBL, TEST_DATA_NBR);
        printf("SGD_CAR_TRAIN\t");
        test_model(sgd_weights, img_mtrx_car_train, len, CAR_LBL, TRAIN_DATA_NBR);
        printf("SGD_TANK_TRAIN\t");
        test_model(sgd_weights, img_mtrx_tank_train, len, TANK_LBL, TRAIN_DATA_NBR);

        printf("\nADAM_CAR_TEST\t");
        test_model(adam_weights, img_mtrx_car_test, len, CAR_LBL, TEST_DATA_NBR);
        printf("ADAM_TANK_TEST\t");
        test_model(adam_weights, img_mtrx_tank_test, len, TANK_LBL, TEST_DATA_NBR);
        printf("ADAM_CAR_TRAIN\t");
        test_model(adam_weights, img_mtrx_car_train, len, CAR_LBL, TRAIN_DATA_NBR);
        printf("ADAM_TANK_TRAIN\t");
        test_model(adam_weights, img_mtrx_tank_train, len, TANK_LBL, TRAIN_DATA_NBR);
        printf("\n");

        write_train_data_to_file(train_data_file_path, loss_arr_GD, loss_arr_SGD, loss_arr_ADAM, time_arr_GD, time_arr_SGD, time_arr_ADAM, EPOCHS);
    }

    // uncomment for test results

    // write epoch/loss time/loss values to file
    // write_train_data_to_file(train_data_file_path, loss_arr_GD, loss_arr_SGD, loss_arr_ADAM, time_arr_GD, time_arr_SGD, time_arr_ADAM, EPOCHS);

    free_matris(img_mtrx_car_train, len, TRAIN_DATA_NBR); // free allocated matrix
    free_matris(img_mtrx_tank_train, len, TRAIN_DATA_NBR);
    free_matris(img_mtrx_car_test, len, TEST_DATA_NBR);
    free_matris(img_mtrx_tank_test, len, TEST_DATA_NBR);

    printf("Code ends\n");
    return 0;
}

void allocate_matris(double ***matris, int width, int height) // allocate memory for matrix
{
    *matris = (double **)malloc(height * sizeof(double *));

    if (*matris == NULL)
    {
        printf("Memory allocation failed for matrix rows.\n");
        exit(1);
    }
    for (int i = 0; i < height; i++)
    {
        (*matris)[i] = (double *)malloc(width * sizeof(double));
        if ((*matris)[i] == NULL)
        {
            printf("Memory allocation failed for matrix columns.\n");
            exit(1);
        }
    }
}

void free_matris(double **matris, int width, int height) // free allocated memory
{

    for (int i = 0; i < height; i++)
    {
        free(matris[i]);
    }

    free(matris);
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

        int index = 0;
        for (int row = 0; row < width; row++)
        {
            for (int col = 0; col < height; col++)
            {

                r = img_data[(row * width + col) * 3];     // Red Channel
                g = img_data[(row * width + col) * 3 + 1]; // Green Channel
                b = img_data[(row * width + col) * 3 + 2]; // Blue Channel

                grayscale = (r + g + b) / 3.0 / 255.0; // Normalize to - [0,1]
                // printf(" index =%d grayscale: %f",index, grayscale);
                x[index] = grayscale;

                // printf(" %d ye %d x= %f\n " , gozlem,index, X[gozlem][index]);

                index++;
            }
        }
        x[index] = 1.0; // add bias

        stbi_image_free(img_data);
    }
    else
    {
        printf("Image read error!!\n");
        exit(1);
    }
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

void fill_parameters(double *vector, double value, int n) // fill up the vector
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

void add_to_fifo(double *fifo, double buffer, int fifo_len) // fifo func for sgd
{
    for (int i = fifo_len; i > 0; i--)
    {
        fifo[i] = fifo[i - 1];
    }
    fifo[0] = buffer;
}

double *gradient_descent_optimization(double *loss_arr, uint64_t *time_arr, double **car_matrix, double **tank_matrix, double init_weight)
{
    static double w[len];
    double y = 0;
    double t;
    double error, loss, gradient, loss_sum;
    uint64_t tick;
    uint64_t tick_c;

    tick = getTick();

    double *x = malloc(len * sizeof(double));
    if (x == NULL)
    {
        printf("Memory allocation failed for x\n");
        exit(1);
    }
    // n*n+1
    fill_parameters(w, init_weight, len); // fill weight vector

    for (int epoch = 1; epoch <= EPOCHS; epoch++)
    {

        loss_sum = 0;

        for (int i = 0; i < (TRAIN_DATA_NBR * 2); i++) // gives images one by one (if i = odd gives Car else Tank)
        {
            if (i % 2 == 0)
            {
                x = &car_matrix[0][(int)(i / 2)];
                t = CAR_LBL;
            }
            else
            {
                x = &tank_matrix[0][(int)(i / 2) - 1];
                t = TANK_LBL;
            }
            y = tanh(dotProduct(x, w, len)); // calculate y_i

            error = y - t;
            loss = error * error;
            loss_sum += loss;
            for (int j = 0; j < len; j++) // main iteration porcess
            {
                gradient = 2 * error * (1 - (y * y)) * x[j]; // calculate gradient
                w[j] = w[j] - lr * gradient;                 // update weights
            }
        }
        tick_c = getTick() - tick;

        if (epoch % EPOCH_PRINT == 0 || epoch == 0)
            printf("GD ||\tepoch(%d/%d) | loss:%lf init_weight: %lf - Tick: %d\n", epoch, EPOCHS, loss_sum / (TRAIN_DATA_NBR * 2), init_weight, tick_c);

        loss_arr[epoch] = loss_sum / (TRAIN_DATA_NBR * 2);
        time_arr[epoch] = tick_c;
#ifdef TEST_INFO
        test_model(w, car_matrix, len, CAR_LBL, TEST_DATA_NBR);
        test_model(w, tank_matrix, len, TANK_LBL, TEST_DATA_NBR);
#endif
    }

    printf("\n");
    return w;
}

double *sgd_optimization(double *loss_arr, uint64_t *time_arr, double **car_matrix, double **tank_matrix, double init_weight)
{
    static double w[len];
    double y = 0;
    double t;
    double error, loss, gradient, loss_sum, gr_sum, loss_sum_bached;
    double gr_arr[BACHES_SGD];
    uint64_t tick;
    uint64_t tick_c;

    double *x = malloc(len * sizeof(double));
    if (x == NULL)
    {
        printf("Memory allocation failed for x\n");
        exit(1);
    }

    fill_parameters(w, init_weight, len);
    tick = getTick();

    for (int epoch = 1; epoch <= EPOCHS; epoch++)
    {

        loss_sum = 0;
        fill_parameters(gr_arr, 0, BACHES_SGD);

        for (int i = 0; i < (TRAIN_DATA_NBR * 2); i++) // gives images one by one (if i = odd gives Car else Tank)
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

            for (int j = 0; j < len; j++)
            {
                gradient = 2 * error * (1 - (y * y)) * x[j]; // accumulate gradients
                add_to_fifo(gr_arr, gradient, BACHES_SGD);

                for (int k = 0; k < BACHES_SGD; k++)
                    gr_sum = gr_arr[k];

                w[j] = w[j] - (lr / BACHES_SGD) * gr_sum; // update weight
            }
        }
        tick_c = getTick() - tick;
        if (epoch % EPOCH_PRINT == 0 || epoch == 0)
            printf("SGD ||\tepoch(%d/%d) | loss:%lf init_weight: %lf Tick:%d\n", epoch, EPOCHS, loss_sum / (TRAIN_DATA_NBR * 2), init_weight, tick_c);

        loss_arr[epoch] = loss_sum / (TRAIN_DATA_NBR * 2);
        time_arr[epoch] = tick_c;
#ifdef TEST_INFO
        test_model(w, car_matrix, len, CAR_LBL, TEST_DATA_NBR);
        test_model(w, tank_matrix, len, TANK_LBL, TEST_DATA_NBR);
#endif
    }
    printf("\n");
    return w;
}

double *adam_optimization(double *loss_arr, uint64_t *time_arr, double **car_matrix, double **tank_matrix, double init_weight)
{

    static double w[len], m[len] = {0}, v[len] = {0};
    double y = 0, t;
    double error, loss, gradient, loss_sum;
    double m_hat = 0, v_hat = 0;
    uint64_t tick;
    uint64_t tick_c;

    double *x = malloc(len * sizeof(double));
    if (x == NULL)
    {
        printf("Memory allocation failed for x\n");
        exit(1);
    }
    // n*n+1
    fill_parameters(w, init_weight, len);

    tick = getTick();

    for (int epoch = 1; epoch <= EPOCHS; epoch++)
    {
        loss_sum = 0;
        for (int i = 0; i < (TRAIN_DATA_NBR * 2); i++) // gives images one by one (if i = odd gives Car else Tank)
        {
            if (i % 2 == 0)
            {
                x = &car_matrix[0][(int)(i / 2)];
                t = CAR_LBL;
            }
            else
            {
                x = &tank_matrix[0][(int)(i / 2) - 1];
                t = TANK_LBL;
            }
            y = tanh(dotProduct(x, w, len)); // calculate y_i

            error = y - t;
            loss = error * error;
            loss_sum += loss;
            for (int j = 0; j < len; j++) // main iteration porcess
            {
                gradient = 2 * error * (1 - (y * y)) * x[j]; // calculate gradient

                m[j] = B1 * m[j] + ((1 - B1) * gradient);              // update m
                v[j] = B2 * v[j] + ((1 - B2) * (gradient * gradient)); // update v

                m_hat = m[j] / (1 - pow(B1, epoch)); // m_hat
                v_hat = v[j] / (1 - pow(B2, epoch)); // v_hat

                w[j] = w[j] - (lr * m_hat / (sqrt(v_hat) + epsilon)); // update weight
            }
        }
        tick_c = getTick() - tick;

        if (epoch % EPOCH_PRINT == 0 || epoch == 0)
            printf("ADAM ||\tepoch(%d/%d) | loss:%lf init_weight: %lf Tick:%d\n", epoch, EPOCHS, loss_sum / (TRAIN_DATA_NBR * 2), init_weight, tick_c);

        loss_arr[epoch] = loss_sum / (TRAIN_DATA_NBR * 2);
        time_arr[epoch] = tick_c;
#ifdef TEST_INFO
        test_model(w, car_matrix, len, CAR_LBL, TEST_DATA_NBR);
        test_model(w, tank_matrix, len, TANK_LBL, TEST_DATA_NBR);
#endif
    }
    printf("\n");
    return w;
}

double model(double *w, double *x, int n) // give weight vector and image vector to model
{
    return tanh(dotProduct(w, x, n));
}

double test_model(double *w, double **x, int n, double lbl, int data_size) // test the model and get truth percentage
{
    double output, true_avg = 0, false_avg = 0;
    int true_predict = 0, false_predict = 0, zero = 0;

    for (int j = 0; j < data_size; j++) // if model > 0 :TANK | else : CAR
    {
        switch ((int)lbl)
        {
        case (int)TANK_LBL:

            output = model(w, x[j], n);
            if (output > 0)
            {
#ifdef INFO
                // printf("TANK Predict True - %lf\n", output);
#endif
                true_predict++;
                true_avg += output;
            }
            else if (output < 0)
            {
#ifdef INFO
                printf("TANK Predict False - %lf j:%d\n", output, j);
#endif
                false_predict++;
                false_avg += output;
            }
            else
            {
#ifdef INFO
                printf("TANK can't predict - %lf j:%d\n", output, j);
#endif
                // print_vector(x[j], n);
                zero++;
            }

            break;

        case (int)CAR_LBL:

            output = model(w, x[j], n);
            if (output < 0)
            {
#ifdef INFO
                // printf("CAR Predict True - %lf\n", output);
#endif
                true_predict++;
                true_avg += output;
            }
            else if (output > 0)
            {
#ifdef INFO
                printf("CAR Predict False - %lf j:%d\n", output, j);

#endif
                false_predict++;
                false_avg += output;
            }
            else
            {
#ifdef INFO
                printf("CAR can't predict - %lf j:%d\n", output, j);
#endif
                // print_vector(x[j], n);
                zero++;
            }

            break;
        }
    }
    true_avg /= true_predict;
    false_avg /= false_predict;

    printf("Predict (%d/%d) Rate:%.3f\t| True avg: %.3lf False_avg: %.3lf zero: %d\n", true_predict, data_size, (double)true_predict / (double)data_size, true_avg, false_avg, zero);
}

void write_train_data_to_file(char *file_name, double gd_loss[EPOCHS], double sgd_loss[EPOCHS], double adam_loss[EPOCHS], uint64_t gd_time[EPOCHS], uint64_t sgd_time[EPOCHS], uint64_t adam_time[EPOCHS], int n)
{
    FILE *fp;
    char string[500] = {"\0"};

    fp = fopen(file_name, "a");

    sprintf(string, "\nepoch,gd_loss,sgd_loss,adam_loss,gd_time,sgd_time,adam_time\n");
    fwrite(string, sizeof(char), strlen(string), fp);

    for (int i = 1; i <= n; i++)
    {
        // epoch,gd_loss,sgd_loss,adam_loss,gd_time,sgd_time,adam_time
        sprintf(string, "%d,%lf,%lf,%lf,%d,%d,%d\n", i, gd_loss[i], sgd_loss[i], adam_loss[i], gd_time[i], sgd_time[i], adam_time[i]);
        fwrite(string, sizeof(char), strlen(string), fp);
    }
    fclose(fp);
    return;
}

void write_weigths_to_file(double w, double *w_gd, double *w_sgd, double *w_adam, int n)
{
    FILE *fp;
    char file_name[100] = {"\0"};
    char string[100] = {"\0"};

    // Write GD weights to file
    sprintf(file_name, "data/weight_%lf_gd.csv", w);
    fp = fopen(file_name, "w");

    sprintf(string, "%lf,", w_gd[0]);
    fwrite(string, sizeof(char), strlen(string), fp);

    for (int i = 1; i < n; i++)
    {
        sprintf(string, ",%lf", w_gd[i]);
        fwrite(string, sizeof(char), strlen(string), fp);
    }
    fclose(fp);

    // Write SGD weights to file
    sprintf(file_name, "data/weight_%lf_sgd.csv", w);
    fp = fopen(file_name, "w");

    sprintf(string, "%lf,", w_sgd[0]);
    fwrite(string, sizeof(char), strlen(string), fp);

    for (int i = 1; i < n; i++)
    {
        sprintf(string, ",%lf", w_sgd[i]);
        fwrite(string, sizeof(char), strlen(string), fp);
    }
    fclose(fp);

    // Write ADAM weights to file
    sprintf(file_name, "data/weight_%lf_adam.csv", w);
    fp = fopen(file_name, "w");

    sprintf(string, "%lf,", w_adam[0]);
    fwrite(string, sizeof(char), strlen(string), fp);

    for (int i = 1; i < n; i++)
    {
        sprintf(string, ",%lf", w_adam[i]);
        fwrite(string, sizeof(char), strlen(string), fp);
    }
    fclose(fp);
}
