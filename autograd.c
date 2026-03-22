#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "autograd.h"
Tensor *tensorAdd(Tensor *a, Tensor *b)
{

	bool req = a->require_grad || b->require_grad;
	Tensor *res = createTensor(a->rows, a->cols, req, false);

	res->forward_fn = add_forward;

	res->indegree = 2;
	res->inTensors = (Tensor **)malloc(sizeof(Tensor *) * 2);
	res->inTensors[0] = a;
	res->inTensors[1] = b;
	if (req)
	{

		res->backward_fn = add_backward;
	}

	return res;
}
Tensor *tensorMatMul(Tensor *a, Tensor *b)
{

	bool req = a->require_grad || b->require_grad;
	Tensor *res = createTensor(a->rows, b->cols, req, false);

	res->forward_fn = matmul_forward;
	res->indegree = 2;
	res->inTensors = (Tensor **)malloc(sizeof(Tensor *) * 2);
	res->inTensors[0] = a;
	res->inTensors[1] = b;

	if (req)
	{

		res->backward_fn = matmul_backward;
	}

	return res;
}
Tensor *tensorSoftmax(Tensor *x)
{
	Tensor *res = createTensor(x->rows, x->cols, x->require_grad, false);

	res->forward_fn = softmax_forward;
	res->indegree = 1;
	res->inTensors = (Tensor **)malloc(sizeof(Tensor *) * 1);
	res->inTensors[0] = x;

	if (x->require_grad)
	{

		res->backward_fn = softmax_backward;
	}

	return res;
}
Tensor *tensorLayerNorm(Tensor *x, Tensor *gamma, Tensor *beta)
{

	Tensor *res = createTensor(x->rows, x->cols, true, false);
	res->forward_fn = layernorm_forward;

	res->indegree = 3;
	res->inTensors = (Tensor **)malloc(sizeof(Tensor *) * 3);
	res->inTensors[0] = x;
	res->inTensors[1] = gamma;
	res->inTensors[2] = beta;
	res->backward_fn = layernorm_backward;

	return res;
}
Tensor *tensorReLU(Tensor *a)
{

	bool req = a->require_grad;
	Tensor *res = createTensor(a->rows, a->cols, req, false);

	res->forward_fn = relu_forward;
	res->indegree = 1;
	res->inTensors = (Tensor **)malloc(sizeof(Tensor *) * 1);
	res->inTensors[0] = a;

	if (req)
	{

		res->backward_fn = relu_backward;
	}

	return res;
}
Tensor *tensorTranspose(Tensor *a)
{

	Tensor *res = createTensor(a->cols, a->rows, a->require_grad, false);

	res->indegree = 1;
	res->inTensors = (Tensor **)malloc(sizeof(Tensor *) * 1);
	res->inTensors[0] = a;

	res->forward_fn = transpose_forward;
	res->backward_fn = transpose_backward;

	return res;
}
Tensor *tensorMulConst(Tensor *a, Tensor *b)
{

	assert(b->rows == 1 && b->cols == 1 && "Second input must be a 1x1 constant tensor");

	bool req = a->require_grad || b->require_grad;
	Tensor *res = createTensor(a->rows, a->cols, req, false);

	res->forward_fn = mul_const_forward;
	res->indegree = 2;
	res->inTensors = (Tensor **)malloc(sizeof(Tensor *) * 2);
	res->inTensors[0] = a;
	res->inTensors[1] = b;

	if (req)
	{
		res->backward_fn = mul_const_backward;
	}

	mul_const_forward(res);

	return res;
}

Tensor *createMaskTensor(int rows, int cols, int *input_ids, int pad_id)
{

	Tensor *mask = createTensor(rows, cols, false, false);

	double neg_inf = -1e9;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			bool should_mask = false;

			if (j > i)
				should_mask = true;

			if (input_ids != NULL && input_ids[j] == pad_id)
				should_mask = true;

			mask->data[i][j] = should_mask ? neg_inf : 0.0;
		}
	}
	return mask;
}
void randomizeTensor(Tensor *t)
{
	// Xavier Initialization
	for (int i = 0; i < t->rows; i++)
	{
		for (int j = 0; j < t->cols; j++)
		{
			double limit = sqrt(6.0 / (t->rows + t->cols));
			t->data[i][j] = ((double)rand() / RAND_MAX) * 2 * limit - limit;
		}
	}
}
Tensor *createTensor(int rows, int cols, bool require_grad, bool is_learnable)
{
	Tensor *newtensor = (Tensor *)malloc(sizeof(Tensor));
	newtensor->rows = rows;
	newtensor->cols = cols;
	newtensor->indegree = 0;
	newtensor->require_grad = require_grad;
	newtensor->is_learnable = is_learnable;

	newtensor->visited = 0;
	int i = 0;
	newtensor->data = (double **)malloc(sizeof(double *) * rows);
	for (i = 0; i < rows; i++)
	{
		newtensor->data[i] = (double *)calloc(cols, sizeof(double));
	}
	newtensor->inTensors = NULL;
	newtensor->forward_fn = NULL;
	newtensor->backward_fn = NULL;
	if (require_grad)
	{
		newtensor->grad = (double **)malloc(sizeof(double *) * rows);
		for (i = 0; i < rows; i++)
		{

			newtensor->grad[i] = (double *)calloc(cols, sizeof(double));
		}
	}
	else
	{
		newtensor->grad = NULL;
	}
	randomizeTensor(newtensor);
	return newtensor;
} // 新建tensor，但是不分配backward,forward
Tensor *createOneHotTensor(int *token_ids, int L, int vsize)
{

	Tensor *res = createTensor(L, vsize, false, false);

	for (int i = 0; i < L; i++)
	{
		int id = token_ids[i];

		assert(id >= 0 && id < vsize && "Token ID out of vocabulary range");

		res->data[i][id] = 1.0;
	}

	return res;
}
Tensor *createPositionalEncoding(int seq_len, int d_model)
{

	Tensor *pe = createTensor(seq_len, d_model, false, false);

	for (int pos = 0; pos < seq_len; pos++)
	{
		for (int i = 0; i < d_model; i += 2)
		{

			double div_term = pow(10000.0, (double)i / (double)d_model);

			pe->data[pos][i] = sin((double)pos / div_term);

			if (i + 1 < d_model)
			{
				pe->data[pos][i + 1] = cos((double)pos / div_term);
			}
		}
	}

	return pe;
}
Tensor *createConstTensor(int rows, int cols, double connum)
{

	Tensor *res = createTensor(rows, cols, 0, false);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			res->data[i][j] = connum;
		}
	}

	return res;
}
void changeOneHotTensor(Tensor *t, int *token_ids, int L, int vsize)
{

	for (int i = 0; i < L; i++)
	{
		for (int j = 0; j < vsize; j++)
		{
			t->data[i][j] = 0.0;
		}
	}

	for (int i = 0; i < L; i++)
	{
		int id = token_ids[i];

		if (id >= 0 && id < vsize)
		{
			t->data[i][id] = 1.0;
		}
		else
		{
			assert(false && "Token ID out of vocabulary range during changeOneHotTensor");
		}
	}
}
void changeMaskTensor(Tensor *t, int *input_ids, int pad_id)
{
	assert(t != NULL && "Target tensor cannot be NULL");

	int rows = t->rows;
	int cols = t->cols;
	double neg_inf = -1e9;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			bool should_mask = false;

			if (j > i)
			{
				should_mask = true;
			}

			if (input_ids != NULL && input_ids[j] == pad_id)
			{
				should_mask = true;
			}

			t->data[i][j] = should_mask ? neg_inf : 0.0;
		}
	}
}

void add_forward(Tensor *t)
{
	Tensor *a = t->inTensors[0];
	Tensor *b = t->inTensors[1];
	for (int i = 0; i < t->rows; i++)
	{
		for (int j = 0; j < t->cols; j++)
		{
			t->data[i][j] = a->data[i][j] + b->data[i][j];
		}
	}
}
void matmul_forward(Tensor *t)
{
	Tensor *a = t->inTensors[0];
	Tensor *b = t->inTensors[1];
	for (int i = 0; i < a->rows; i++)
	{
		for (int j = 0; j < b->cols; j++)
		{
			double sum = 0;
			for (int k = 0; k < a->cols; k++)
			{
				sum += a->data[i][k] * b->data[k][j];
			}
			t->data[i][j] = sum;
		}
	}
}
void relu_forward(Tensor *t)
{
	Tensor *a = t->inTensors[0];
	for (int i = 0; i < t->rows; i++)
	{
		for (int j = 0; j < t->cols; j++)
		{
			t->data[i][j] = (a->data[i][j] > 0) ? a->data[i][j] : 0.0;
		}
	}
}
void layernorm_forward(Tensor *t)
{
	Tensor *x = t->inTensors[0];
	Tensor *gamma = t->inTensors[1];
	Tensor *beta = t->inTensors[2];
	double eps = 1e-5;

	for (int i = 0; i < x->rows; i++)
	{
		double sum = 0, sq_sum = 0;
		for (int j = 0; j < x->cols; j++)
		{
			sum += x->data[i][j];
			sq_sum += x->data[i][j] * x->data[i][j];
		}
		double mu = sum / x->cols;
		double var = sq_sum / x->cols - mu * mu;
		if (var < 0)
			var = 0;
		double std_inv = 1.0 / sqrt(var + eps);

		for (int j = 0; j < x->cols; j++)
		{
			double x_hat = (x->data[i][j] - mu) * std_inv;
			t->data[i][j] = x_hat * gamma->data[0][j] + beta->data[0][j];
		}
	}
}
void softmax_forward(Tensor *t)
{
	Tensor *x = t->inTensors[0];
	for (int i = 0; i < x->rows; i++)
	{
		double max_val = x->data[i][0];
		for (int j = 1; j < x->cols; j++)
			if (x->data[i][j] > max_val)
				max_val = x->data[i][j];

		double sum = 0;
		for (int j = 0; j < x->cols; j++)
		{
			t->data[i][j] = exp(x->data[i][j] - max_val);
			sum += t->data[i][j];
		}
		for (int j = 0; j < x->cols; j++)
			t->data[i][j] /= sum;
	}
}
void softmax_then_cross_entropy_forward(Tensor *t)
{
	Tensor *logits = t->inTensors[0];
	Tensor *target = t->inTensors[1];
	int rows = logits->rows;
	int cols = logits->cols;
	double total_loss = 0.0;

	for (int i = 0; i < rows; i++)
	{
		double max_val = logits->data[i][0];
		for (int j = 1; j < cols; j++)
			if (logits->data[i][j] > max_val)
				max_val = logits->data[i][j];

		double sum_exp = 0;
		for (int j = 0; j < cols; j++)
			sum_exp += exp(logits->data[i][j] - max_val);

		for (int j = 0; j < cols; j++)
		{
			if (target->data[i][j] > 0)
			{
				double prob = exp(logits->data[i][j] - max_val) / sum_exp;
				if (prob < 1e-12)
					prob = 1e-12;
				total_loss -= target->data[i][j] * log(prob);
			}
		}
	}
	t->data[0][0] = total_loss / (double)rows;
}
void transpose_forward(Tensor *t)
{
	Tensor *a = t->inTensors[0];
	for (int i = 0; i < a->rows; i++)
	{
		for (int j = 0; j < a->cols; j++)
		{
			// t->rows 是 a->cols, t->cols 是 a->rows
			t->data[j][i] = a->data[i][j];
		}
	}
}
void mul_const_forward(Tensor *t)
{

	Tensor *a = t->inTensors[0];
	Tensor *b = t->inTensors[1];

	double const_val = b->data[0][0];

	for (int i = 0; i < a->rows; i++)
	{
		for (int j = 0; j < a->cols; j++)
		{
			t->data[i][j] = a->data[i][j] * const_val;
		}
	}
}
void forward(Tensor *loss)
{
	Tensor *topo_order[1024];
	int node_count = 0;
	flush_visit(loss);
	build_topo(loss, topo_order, &node_count);

	for (int i = 0; i < node_count; i++)
	{
		if (topo_order[i]->forward_fn != NULL)
		{
			topo_order[i]->forward_fn(topo_order[i]);
		}
	}
}

void build_topo(Tensor *t, Tensor **topo_array, int *index)
{
	// 获得计算图的拓扑排序，存入topoarray
	if (t == NULL || t->visited)
		return;
	t->visited = true;
	for (int i = 0; i < t->indegree; i++)
	{
		build_topo(t->inTensors[i], topo_array, index);
	}
	// 当所有依赖项都处理完后，把当前节点加入序列
	// 此时序列的顺序是：[输入层, ... 隐藏层, ... 输出层/Loss]
	topo_array[(*index)++] = t;
}
void flush_visit(Tensor *t)
{
	if (t == NULL || t->visited == false)
	{
		return;
	}
	t->visited = false;
	for (int i = 0; i < t->indegree; i++)
	{
		flush_visit(t->inTensors[i]);
	}
}
void backward(Tensor *loss)
{

	Tensor *topo_order[1024];
	int node_count = 0;
	flush_visit(loss);
	build_topo(loss, topo_order, &node_count);
	if (loss->require_grad && loss->grad != NULL)
	{
		loss->grad[0][0] = 1.0;
	}
	for (int i = node_count - 1; i >= 0; i--)
	{
		Tensor *curr = topo_order[i];
		// 按照拓扑的逆序backward，确保所有梯度到位之后才启动backward
		if (curr->backward_fn != NULL && curr->indegree > 0)
		{
			curr->backward_fn(curr);
		}
	}
}
void matmul_backward(Tensor *t)
{
	Tensor *a = t->inTensors[0];
	Tensor *b = t->inTensors[1];

	// dA = dC * B^T
	if (a->require_grad)
	{
		for (int i = 0; i < a->rows; i++)
		{
			for (int j = 0; j < a->cols; j++)
			{
				double sum = 0;
				for (int k = 0; k < t->cols; k++)
				{
					// t->grad[i][k] 是 dC
					// b->data[j][k] 是 B^T 的元素 (原本是 B[k][j])
					sum += t->grad[i][k] * b->data[j][k];
				}
				a->grad[i][j] += sum;
			}
		}
	}

	// dB = A^T * dC
	if (b->require_grad)
	{
		for (int i = 0; i < b->rows; i++)
		{
			for (int j = 0; j < b->cols; j++)
			{
				double sum = 0;
				for (int k = 0; k < a->rows; k++)
				{
					// a->data[k][i] 是 A^T 的元素 (原本是 A[i][k])
					// t->grad[k][j] 是 dC
					sum += a->data[k][i] * t->grad[k][j];
				}
				b->grad[i][j] += sum;
			}
		}
	}
}
void add_backward(Tensor *t)
{

	Tensor *a = t->inTensors[0];
	Tensor *b = t->inTensors[1];

	assert(a->rows == b->rows && a->cols == b->cols);
	assert(t->rows == a->rows && t->cols == a->cols);
	if (t->grad == NULL)
		return;

	for (int i = 0; i < t->rows; i++)
	{
		for (int j = 0; j < t->cols; j++)
		{

			if (a->require_grad && a->grad != NULL)
			{
				a->grad[i][j] += t->grad[i][j];
			}
			if (b->require_grad && b->grad != NULL)
			{
				b->grad[i][j] += t->grad[i][j];
			}
		}
	}
}
// update函数执行后会清空所有的visit

void relu_backward(Tensor *t)
{

	Tensor *a = t->inTensors[0];

	if (!a->require_grad || a->grad == NULL)
		return;

	for (int i = 0; i < t->rows; i++)
	{
		for (int j = 0; j < t->cols; j++)
		{

			if (a->data[i][j] > 0)
			{
				a->grad[i][j] += t->grad[i][j];
			}
		}
	}
}
void layernorm_backward(Tensor *t)
{
	// t->inTensors[0]: X (Input, rows x cols)
	// t->inTensors[1]: Gamma (Learnable, 1 x cols)
	// t->inTensors[2]: Beta (Learnable, 1 x cols)
	Tensor *x = t->inTensors[0];
	Tensor *gamma = t->inTensors[1];
	Tensor *beta = t->inTensors[2];

	int R = t->rows;
	int C = t->cols;
	double eps = 1e-5;

	for (int i = 0; i < R; i++)
	{

		double sum = 0, sq_sum = 0;
		for (int j = 0; j < C; j++)
		{
			sum += x->data[i][j];
			sq_sum += x->data[i][j] * x->data[i][j];
		}
		double mu = sum / C;
		double var = sq_sum / C - mu * mu;
		double std_inv = 1.0 / sqrt(var + eps);

		double dl_dvar = 0;
		double dl_dmu = 0;

		for (int j = 0; j < C; j++)
		{
			double x_hat = (x->data[i][j] - mu) * std_inv;
			double dy = t->grad[i][j];

			if (gamma->require_grad)
				gamma->grad[0][j] += dy * x_hat;
			if (beta->require_grad)
				beta->grad[0][j] += dy;

			double dl_dxhat = dy * gamma->data[0][j];
			dl_dvar += dl_dxhat * (x->data[i][j] - mu) * -0.5 * pow(std_inv, 3);
			dl_dmu += dl_dxhat * (-std_inv);
		}

		if (x->require_grad)
		{
			for (int j = 0; j < C; j++)
			{
				double dl_dxhat = t->grad[i][j] * gamma->data[0][j];
				x->grad[i][j] += dl_dxhat * std_inv +
								 dl_dvar * 2.0 * (x->data[i][j] - mu) / C +
								 dl_dmu / C;
			}
		}
	}
}
void softmax_backward(Tensor *t)
{

	Tensor *x = t->inTensors[0];

	if (!x->require_grad || x->grad == NULL)
		return;

	int R = t->rows;
	int C = t->cols;

	for (int i = 0; i < R; i++)
	{

		double sum_dy_y = 0;
		for (int j = 0; j < C; j++)
		{

			sum_dy_y += t->grad[i][j] * t->data[i][j];
		}

		for (int j = 0; j < C; j++)
		{
			double yi = t->data[i][j];
			double dyi = t->grad[i][j];
			x->grad[i][j] += yi * (dyi - sum_dy_y);
		}
	}
}
void softmax_then_cross_entropy_backward(Tensor *t)
{

	Tensor *logits = t->inTensors[0];
	Tensor *target = t->inTensors[1];

	if (!logits->require_grad || logits->grad == NULL)
		return;

	int rows = logits->rows;
	int cols = logits->cols;

	for (int i = 0; i < rows; i++)
	{

		double max_val = -1e9;
		for (int j = 0; j < cols; j++)
			if (logits->data[i][j] > max_val)
				max_val = logits->data[i][j];

		double sum_exp = 0;
		for (int j = 0; j < cols; j++)
			sum_exp += exp(logits->data[i][j] - max_val);

		for (int j = 0; j < cols; j++)
		{
			double prob = exp(logits->data[i][j] - max_val) / sum_exp;
			double t_ij = target->data[i][j];

			logits->grad[i][j] += (prob - t_ij) / (double)rows;
		}
	}
}
Tensor *tensorSoftmaxThenCrossEntropy(Tensor *logits, Tensor *target)
{
	assert(logits->rows == target->rows && logits->cols == target->cols);

	Tensor *res = createTensor(1, 1, true, false);
	res->forward_fn = softmax_then_cross_entropy_forward;

	res->indegree = 2;
	res->inTensors = (Tensor **)malloc(sizeof(Tensor *) * 2);
	res->inTensors[0] = logits;
	res->inTensors[1] = target;
	res->backward_fn = softmax_then_cross_entropy_backward;
	return res;
}
void transpose_backward(Tensor *t)
{
	Tensor *a = t->inTensors[0];
	if (!a->require_grad || a->grad == NULL)
		return;

	for (int i = 0; i < a->rows; i++)
	{
		for (int j = 0; j < a->cols; j++)
		{

			a->grad[i][j] += t->grad[j][i];
		}
	}
}
void mul_const_backward(Tensor *t)
{
	Tensor *a = t->inTensors[0];
	Tensor *b = t->inTensors[1];
	double const_val = b->data[0][0];

	for (int i = 0; i < t->rows; i++)
	{
		for (int j = 0; j < t->cols; j++)
		{

			if (a->require_grad && a->grad != NULL)
			{
				a->grad[i][j] += t->grad[i][j] * const_val;
			}

			if (b->require_grad && b->grad != NULL)
			{
				b->grad[0][0] += t->grad[i][j] * a->data[i][j];
			}
		}
	}
}

void update(Tensor *loss, double lr)
{
	Tensor *topo_order[1024];
	int node_count = 0;
	flush_visit(loss);
	build_topo(loss, topo_order, &node_count);
	for (int i = 0; i < node_count; i++)
	{
		Tensor *t = topo_order[i];

		if (t->is_learnable && t->require_grad && t->grad != NULL)
		{
			for (int r = 0; r < t->rows; r++)
			{
				for (int c = 0; c < t->cols; c++)
				{
					t->data[r][c] -= lr * t->grad[r][c];
				}
			}
		}

		if (t->require_grad && t->grad != NULL)
		{
			for (int r = 0; r < t->rows; r++)
			{
				for (int c = 0; c < t->cols; c++)
				{
					t->grad[r][c] = 0.0;
				}
			}
		}
	}

	flush_visit(loss);
}

void freeTensor(Tensor *t)
{
	if (t == NULL)
		return;

	if (t->data != NULL)
	{
		for (int i = 0; i < t->rows; i++)
		{
			if (t->data[i] != NULL)
			{
				free(t->data[i]);
			}
		}
		free(t->data);
	}

	if (t->grad != NULL)
	{
		for (int i = 0; i < t->rows; i++)
		{
			if (t->grad[i] != NULL)
			{
				free(t->grad[i]);
			}
		}
		free(t->grad);
	}
	if (t->inTensors != NULL)
	{
		free(t->inTensors);
	}
	free(t);
}
// 释放以 loss 节点为起点的整个计算图

void freeGraph(Tensor *loss)
{
	if (loss == NULL)
		return;

	Tensor *topo_order[1024]; 
	int node_count = 0;

	flush_visit(loss);

	build_topo(loss, topo_order, &node_count);

	for (int i = 0; i < node_count; i++)
	{
		freeTensor(topo_order[i]);
	}
}
void add_bias_forward(Tensor *t)
{
	Tensor *a = t->inTensors[0];
	Tensor *bias = t->inTensors[1];

	for (int i = 0; i < t->rows; i++)
	{
		for (int j = 0; j < t->cols; j++)
		{
		
			t->data[i][j] = a->data[i][j] + bias->data[0][j];
		}
	}
}


void add_bias_backward(Tensor *t)
{
	Tensor *a = t->inTensors[0];
	Tensor *bias = t->inTensors[1];

	for (int i = 0; i < t->rows; i++)
	{
		for (int j = 0; j < t->cols; j++)
		{
			
			if (a->require_grad && a->grad != NULL)
			{
				a->grad[i][j] += t->grad[i][j];
			}
			
			if (bias->require_grad && bias->grad != NULL)
			{
				bias->grad[0][j] += t->grad[i][j];
			}
		}
	}
}


Tensor *tensorAddBias(Tensor *a, Tensor *bias)
{
	
	if (bias->rows != 1 || bias->cols != a->cols)
	{
		printf("Error: tensorAddBias dimension mismatch!\n");
		exit(1);
	}

	bool req = a->require_grad || bias->require_grad;
	Tensor *res = createTensor(a->rows, a->cols, req, false);

	res->forward_fn = add_bias_forward;
	res->indegree = 2;
	res->inTensors = (Tensor **)malloc(sizeof(Tensor *) * 2);
	res->inTensors[0] = a;
	res->inTensors[1] = bias;

	if (req)
	{
		res->backward_fn = add_bias_backward;
	}

	return res;
}
