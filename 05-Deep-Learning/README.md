# Deep Learning Complete -- From StatQuest to Exam-Ready

A comprehensive, exam-ready deep learning curriculum built from StatQuest lectures (L01-L33). Each directory combines a Theory.md (rigorous, exam-ready reference) and a lab.py (hands-on runnable code) covering one major deep learning topic. The material progresses from fundamental neurons all the way to reinforcement learning from human feedback (RLHF) and a ChatGPT-like transformer implementation in PyTorch.

This course is designed to be **the single reference you need** for deep learning interviews, exams, or practical coding. Every Theory.md follows a consistent format: Motivation and Intuition, Formal Definition with formulas, Term tables, Worked Numerical Examples, Python code blocks, and bold ML Connection callouts. Every lab.py is self-contained and runnable, with print statements at every step to make tensor shapes and intermediate values visible.

## Course Map

| Lecture(s) | Topic Directory | Topics Covered |
| :--- | :--- | :--- |
| L01-L07 | 01-Neural-Network-Fundamentals | Perceptron, forward pass, chain rule, gradient descent, backpropagation (main ideas and full details) |
| L08 | 02-Activation-Functions | ReLU, sigmoid, tanh, ELU, leaky ReLU, dying ReLU problem, activation function derivatives |
| L09-L13 | 03-Multi-Output-and-Loss-Functions | Softmax, cross-entropy, argmax, multiple outputs, softmax derivative step-by-step, cross-entropy backpropagation |
| L14 | 04-Convolutional-Neural-Networks | Convolution operation, pooling, padding/strides, LeNet, AlexNet, VGGNet, ResNet architectures |
| L15-L16 | 05-Recurrent-Architectures | RNN, LSTM (forget gate, input gate, output gate, cell state), GRU, backpropagation through time (BPTT), vanishing gradient |
| L17 | 06-Word-Embeddings | Word2Vec (CBOW, skip-gram), GloVe, FastText, embedding analogies (king - man + woman = queen) |
| L18-L19 | 07-Seq2Seq-and-Attention | Encoder-decoder architecture, teacher forcing, attention mechanism (Bahdanau, Luong), context vectors |
| L20-L22 | 08-Transformers-GPT-BERT | Self-attention, multi-head attention, positional encoding, GPT decoder-only, BERT encoder-only, masked LM, next-sentence prediction |
| L23-L25 | 09-Reinforcement-Learning | Q-learning, deep Q-networks, policy gradients, REINFORCE, actor-critic, RLHF (reward model, PPO) |
| L26-L28 | 10-Advanced-Math-for-DL | Tensors (0D-4D), matrix algebra for forward/backward pass, broadcasting, attention math, einsum, multi-head attention shapes, gradient outer-product form |
| L29-L33 | 11-PyTorch-Projects | PyTorch tensors and autograd, nn.Module, training loop, PyTorch Lightning, LSTM for sequence prediction, word embeddings, ChatGPT-like decoder-only transformer |

## Recommended Learning Path

The course is designed to be taken **sequentially** -- each topic builds on the previous:

1. **01-Neural-Network-Fundamentals** -- Start here. Every deep learning concept traces back to the perceptron, forward pass, chain rule, gradient descent, and backpropagation.
2. **02-Activation-Functions** -- Add nonlinearity to your networks. Understand why ReLU dominates and when to use alternatives.
3. **03-Multi-Output-and-Loss-Functions** -- Move beyond binary classification. Softmax + cross-entropy is the standard for multi-class problems.
4. **04-Convolutional-Neural-Networks** -- The standard architecture for image data. Convolutions, pooling, and classic architectures from LeNet to ResNet.
5. **05-Recurrent-Architectures** -- Sequence modeling with RNNs and LSTMs. Understand the vanishing gradient problem and how gates solve it.
6. **06-Word-Embeddings** -- From one-hot to dense vectors. Word2Vec skip-gram with negative sampling, and the famous analogy examples.
7. **07-Seq2Seq-and-Attention** -- The encoder-decoder framework and the attention mechanism that revolutionized NLP.
8. **08-Transformers-GPT-BERT** -- The modern backbone of NLP and beyond. Self-attention, multi-head, positional encoding, and the GPT/BERT architectures.
9. **09-Reinforcement-Learning** -- From Q-learning to RLHF. Understand how ChatGPT was trained with human feedback.
10. **10-Advanced-Math-for-DL** -- The mathematical foundation underlying all of the above. Tensors, matrix algebra, attention math, einsum, and gradient flow.
11. **11-PyTorch-Projects** -- Put it all into practice. Five projects from scratch: tensor ops, regression NN, LSTM forecasting, word embeddings, and a mini ChatGPT-like transformer for character generation.

## Philosophy

These materials are "TB-final-exam ready" -- meaning they are designed for last-mile revision before technical interviews, graduate exams, or practical coding assessments. Each file is self-contained, rigorous, and structured for rapid lookup. The emphasis is on:

- **Exam-ready formatting** -- numbered sections, term definition tables after every formula, worked examples, Python code blocks, and intuition-first exposition.
- **Runnable code** -- every lab.py runs end-to-end with no missing dependencies beyond torch and numpy.
- **Shape-aware programming** -- every tensor operation in lab.py prints its shape so you can trace the dimensions through the computation.
- **Breadth-to-depth** -- each topic covers the full range from first principles to cutting-edge architecture.
