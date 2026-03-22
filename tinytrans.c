#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "autograd.h"


#define D_MODEL 64
#define D_FF 128
#define NUM_EPOCHS 80 


#define LR 0.003
#define MAX_GEN_LEN 200
#define TEMPERATURE 1.0


void shuffle_indices(int *array, int n)
{
    for (int i = n - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

char *my_strdup(const char *s)
{
    char *d = (char *)malloc(strlen(s) + 1);
    if (d != NULL)
    {
        strcpy(d, s);
    }
    return d;
}

int main()
{
    srand((unsigned int)time(NULL));

    FILE *fp = fopen("test.txt", "r");
    if (!fp)
    {
        perror("cant open test.txt");
        return 1;
    }

    char **all_words = NULL;
    int num_words = 0;
    char buffer[256];
    while (fscanf(fp, "%s", buffer) == 1)
    {
        all_words = (char **)realloc(all_words, (num_words + 1) * sizeof(char *));
        all_words[num_words] = my_strdup(buffer);
        num_words++;
    }
    fclose(fp);

    if (num_words < 35)
    {
        printf("too few words\n");
        return 1;
    }

 
    char **vocab = NULL;
    int vocab_size = 0;
    int *token_ids = (int *)malloc(num_words * sizeof(int));
    for (int i = 0; i < num_words; i++)
    {
        bool found = false;
        for (int v = 0; v < vocab_size; v++)
        {
            if (strcmp(vocab[v], all_words[i]) == 0)
            {
                token_ids[i] = v;
                found = true;
                break;
            }
        }
        if (!found)
        {
            vocab = (char **)realloc(vocab, (vocab_size + 1) * sizeof(char *));
            vocab[vocab_size] = my_strdup(all_words[i]);
            token_ids[i] = vocab_size;
            vocab_size++;
        }
    }

    int seq_len = 8;
    int *input_ids = (int *)malloc(seq_len * sizeof(int));
    int *target_ids = (int *)malloc(seq_len * sizeof(int));

   
    int max_samples = num_words - seq_len; 
    int *sample_start_indices = (int *)malloc(max_samples * sizeof(int));
    for (int i = 0; i < max_samples; i++)
    {
        sample_start_indices[i] = i;
    }
    // ------------------------------------


    Tensor *input_onehot = createTensor(seq_len, vocab_size, false, false);
    Tensor *target_onehot = createTensor(seq_len, vocab_size, false, false);
    Tensor *mask = createMaskTensor(seq_len, seq_len, NULL, -1);

    Tensor *W_emb = createTensor(vocab_size, D_MODEL, true, true);
    Tensor *Wq = createTensor(D_MODEL, D_MODEL, true, true);
    Tensor *Wk = createTensor(D_MODEL, D_MODEL, true, true);
    Tensor *Wv = createTensor(D_MODEL, D_MODEL, true, true);

    Tensor *gamma1 = createTensor(1, D_MODEL, true, true);
    Tensor *beta1 = createTensor(1, D_MODEL, true, true);
    for (int j = 0; j < D_MODEL; j++) { gamma1->data[0][j] = 1.0; beta1->data[0][j] = 0.0; }

    Tensor *W1 = createTensor(D_MODEL, D_FF, true, true);
    Tensor *W2 = createTensor(D_FF, D_MODEL, true, true);

    Tensor *gamma2 = createTensor(1, D_MODEL, true, true);
    Tensor *beta2 = createTensor(1, D_MODEL, true, true);
    for (int j = 0; j < D_MODEL; j++) { gamma2->data[0][j] = 1.0; beta2->data[0][j] = 0.0; }

    Tensor *W_out = createTensor(D_MODEL, vocab_size, true, true);


    Tensor *embed = tensorMatMul(input_onehot, W_emb);
    Tensor *pe = createPositionalEncoding(seq_len, D_MODEL);
    Tensor *x = tensorAdd(embed, pe);

    Tensor *q = tensorMatMul(x, Wq);
    Tensor *k = tensorMatMul(x, Wk);
    Tensor *v = tensorMatMul(x, Wv);
    Tensor *kt = tensorTranspose(k);
    Tensor *scores = tensorMatMul(q, kt);
    Tensor *scale_const = createConstTensor(1, 1, 1.0 / sqrt((double)D_MODEL));
    Tensor *scaled_scores = tensorMulConst(scores, scale_const);
    Tensor *masked_scores = tensorAdd(scaled_scores, mask);
    Tensor *attn_w = tensorSoftmax(masked_scores);
    Tensor *context = tensorMatMul(attn_w, v);
    Tensor *attn_out = tensorAdd(x, context);
    Tensor *norm1 = tensorLayerNorm(attn_out, gamma1, beta1);

    Tensor *ffn_inter = tensorMatMul(norm1, W1);
    Tensor *relu_out = tensorReLU(ffn_inter);
    Tensor *ffn_out = tensorMatMul(relu_out, W2);
    Tensor *ffn_residual = tensorAdd(norm1, ffn_out);
    Tensor *norm2 = tensorLayerNorm(ffn_residual, gamma2, beta2);

    Tensor *logits = tensorMatMul(norm2, W_out);
    Tensor *loss = tensorSoftmaxThenCrossEntropy(logits, target_onehot);
    int total=NUM_EPOCHS;
    printf(" Tiny Transformer training(Vocab: %d, Samples: %d),we will train for %d EPOCHS.this might cost minutes\n", vocab_size, max_samples,total);
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++)
    {
        double epoch_loss = 0;
        int steps = 0;

        shuffle_indices(sample_start_indices, max_samples);

        for (int s = 0; s < max_samples; s++)
        {
            int start_idx = sample_start_indices[s];

            for (int i = 0; i < seq_len; i++)
            {
                input_ids[i] = token_ids[start_idx + i];
                target_ids[i] = token_ids[start_idx + i + 1];
            }
            changeOneHotTensor(input_onehot, input_ids, seq_len, vocab_size);
            changeOneHotTensor(target_onehot, target_ids, seq_len, vocab_size);

            forward(loss);
            epoch_loss += loss->data[0][0];
            steps++;
            backward(loss);
            update(loss, LR);
        }
        if (epoch % 10 == 0)
            printf("Epoch %d | Avg Loss: %.4f\n", epoch, epoch_loss / steps);
    }

    printf("\n=== test ===\n");
    int *gen_ids = (int *)malloc(seq_len * sizeof(int));
    for (int i = 0; i < seq_len; i++) gen_ids[i] = token_ids[i]; 

    printf(" Seed context: ");
    for (int i = 0; i < seq_len; i++) printf("%s ", vocab[gen_ids[i]]);
    printf("\ngenerated: ");

    float *probs = (float *)malloc(vocab_size * sizeof(float));
    for (int step = 0; step < MAX_GEN_LEN; step++)
    {
        changeOneHotTensor(input_onehot, gen_ids, seq_len, vocab_size);
        forward(logits);

        int predict_idx = seq_len - 1;
        float max_logit = -1e9;
        for (int v = 0; v < vocab_size; v++)
            if (logits->data[predict_idx][v] > max_logit) max_logit = (float)logits->data[predict_idx][v];

        float sum_exp = 0.0f;
        for (int v = 0; v < vocab_size; v++)
        {
            probs[v] = expf((float)((logits->data[predict_idx][v] - max_logit) / TEMPERATURE));
            sum_exp += probs[v];
        }
        for (int v = 0; v < vocab_size; v++) probs[v] /= sum_exp;

        float r = (float)rand() / RAND_MAX;
        float cumulative = 0.0f;
        int next_token = vocab_size - 1;
        for (int v = 0; v < vocab_size; v++)
        {
            cumulative += probs[v];
            if (r <= cumulative) { next_token = v; break; }
        }

        printf("%s ", vocab[next_token]);
        memmove(gen_ids, gen_ids + 1, (seq_len - 1) * sizeof(int));
        gen_ids[seq_len - 1] = next_token;
    }
    printf("\n============================\n");


    free(probs); freeGraph(loss);
    for (int i = 0; i < num_words; i++) free(all_words[i]); free(all_words);
    for (int i = 0; i < vocab_size; i++) free(vocab[i]); free(vocab);
    free(token_ids); free(input_ids); free(target_ids); free(gen_ids); free(sample_start_indices);

    return 0;
}