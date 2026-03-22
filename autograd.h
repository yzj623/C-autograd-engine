typedef struct tensor
{
	int rows; // 行
	int cols; // 列
	int indegree;
	double **data;
	double **grad;
	struct tensor **inTensors;
	void (*forward_fn)(struct tensor *self);
	void (*backward_fn)(struct tensor *self);
	bool require_grad;
	bool is_learnable;
	bool visited;
} Tensor;
// 基础管理与生命周期
void randomizeTensor(Tensor *t);
Tensor *createTensor(int rows, int cols, bool require_grad, bool is_learnable);
Tensor *createMaskTensor(int rows, int cols, int *input_ids, int pad_id);

// 计算图构建与拓扑排序
void build_topo(Tensor *t, Tensor **topo_array, int *index);
void flush_visit(Tensor *t);

// 自动微分与优化
void forward(Tensor *loss);
void backward(Tensor *loss);
void update(Tensor *loss, double lr);

// 矩阵加法
void add_forward(Tensor *t);
void add_backward(Tensor *t);
Tensor *tensorAdd(Tensor *a, Tensor *b);
void add_bias_forward(Tensor *t);
void add_bias_backward(Tensor *t);
Tensor *tensorAddBias(Tensor *a, Tensor *bias);

// 矩阵乘法
void matmul_forward(Tensor *t);
void matmul_backward(Tensor *t);
Tensor *tensorMatMul(Tensor *a, Tensor *b);

// ReLU 激活函数
void relu_forward(Tensor *t);
void relu_backward(Tensor *t);
Tensor *tensorReLU(Tensor *a);

// Layer Norm
void layernorm_forward(Tensor *t);
void layernorm_backward(Tensor *t);
Tensor *tensorLayerNorm(Tensor *x, Tensor *gamma, Tensor *beta);

// Softmax
void softmax_forward(Tensor *t);
void softmax_backward(Tensor *t);
Tensor *tensorSoftmax(Tensor *x);

// 交叉熵损失
void softmax_then_cross_entropy_forward(Tensor *t);
void softmax_then_cross_entropy_backward(Tensor *t);
Tensor *tensorSoftmaxThenCrossEntropy(Tensor *logits, Tensor *target);
// else
Tensor *tensorTranspose(Tensor *a);
void transpose_backward(Tensor *t);
void transpose_forward(Tensor *t);
Tensor *createOneHotTensor(int *token_ids, int L, int vsize);
Tensor *createPositionalEncoding(int seq_len, int d_model);
void changeOneHotTensor(Tensor *t, int *token_ids, int L, int vsize);
void changeMaskTensor(Tensor *t, int *input_ids, int pad_id);
void mul_const_forward(Tensor *t);
void mul_const_backward(Tensor *t);
Tensor *tensorMulConst(Tensor *a, Tensor *b);
Tensor *createConstTensor(int rows, int cols, double connum);
void freeTensor(Tensor *t);
void freeGraph(Tensor *loss);
