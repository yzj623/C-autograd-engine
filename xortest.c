#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include "autograd.h"

#define BATCH_SIZE 4
#define HIDDEN_SIZE 8
#define NUM_EPOCHS 3000
#define LR 0.15

int main()
{

    srand((unsigned int)time(NULL));

    Tensor *x_train = createTensor(BATCH_SIZE, 2, false, false);
    double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 2; j++)
            x_train->data[i][j] = inputs[i][j];

    Tensor *y_train = createTensor(BATCH_SIZE, 2, false, false);
    double targets[4][2] = {{1, 0}, {0, 1}, {0, 1}, {1, 0}};
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 2; j++)
            y_train->data[i][j] = targets[i][j];

    Tensor *W1 = createTensor(2, HIDDEN_SIZE, true, true);
    Tensor *b1 = createTensor(1, HIDDEN_SIZE, true, true);
    for (int j = 0; j < HIDDEN_SIZE; j++)
        b1->data[0][j] = 0.01;

    Tensor *W2 = createTensor(HIDDEN_SIZE, 2, true, true);
    Tensor *b2 = createTensor(1, 2, true, true);
    for (int j = 0; j < 2; j++)
        b2->data[0][j] = 0.01;

    // Hidden = ReLU(X * W1 + b1)
    Tensor *layer1_matmul = tensorMatMul(x_train, W1);
    Tensor *layer1_add = tensorAddBias(layer1_matmul, b1);
    Tensor *layer1_act = tensorReLU(layer1_add);

    // Logits = Hidden * W2 + b2
    Tensor *layer2_matmul = tensorMatMul(layer1_act, W2);
    Tensor *logits = tensorAddBias(layer2_matmul, b2);

    // Loss = Softmax + Cross Entropy
    Tensor *loss = tensorSoftmaxThenCrossEntropy(logits, y_train);

    printf("=== Robust XOR Training Starting ===\n");
    for (int epoch = 0; epoch <= NUM_EPOCHS; epoch++)
    {
        forward(loss);

        if (epoch % 500 == 0)
        {
            printf("Epoch %4d | Loss: %.6f\n", epoch, loss->data[0][0]);
        }

        backward(loss);
        update(loss, LR);
    }

    printf("\n=== Final Test Results ===\n");
    forward(logits);

    int success_count = 0;
    for (int i = 0; i < 4; i++)
    {
        int pred = (logits->data[i][1] > logits->data[i][0]) ? 1 : 0;
        int target = (targets[i][1] > targets[i][0]) ? 1 : 0;

        double sum_exp = exp(logits->data[i][0]) + exp(logits->data[i][1]);
        double p1 = exp(logits->data[i][1]) / sum_exp;

        printf("In: [%.0f, %.0f] | Predict: %d | Target: %d | Prob(1): %.4f\n",
               x_train->data[i][0], x_train->data[i][1], pred, target, p1);

        if (pred == target)
            success_count++;
    }

    if (success_count == 4)
        printf("\n[Success] Model solved XOR perfectly!\n");
    else
        printf("\n[Partial] Model struggle with some cases.\n");

   
    freeGraph(loss);
    return 0;
}
