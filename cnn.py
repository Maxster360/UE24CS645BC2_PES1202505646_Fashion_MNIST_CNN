import numpy as np


class ConvolutionLayer:
    """Convolutional layer implementing forward and backward pass from scratch."""

    def __init__(self, num_filters, filter_size, in_channels=1, stride=1, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        self.filters = np.random.randn(
            num_filters, in_channels, filter_size, filter_size
        ) * np.sqrt(2.0 / (in_channels * filter_size * filter_size))
        self.biases = np.zeros(num_filters)

    def _pad_input(self, input_data):
        if self.padding > 0:
            if input_data.ndim == 2:
                return np.pad(
                    input_data,
                    ((self.padding, self.padding), (self.padding, self.padding)),
                    mode="constant",
                )
            else:
                return np.pad(
                    input_data,
                    ((0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                    mode="constant",
                )
        return input_data

    def forward(self, input_data):
        if input_data.ndim == 2:
            input_data = input_data[np.newaxis, :, :]
        self.input_data = input_data
        self.padded_input = self._pad_input(input_data)
        _, h, w = self.padded_input.shape
        out_h = (h - self.filter_size) // self.stride + 1
        out_w = (w - self.filter_size) // self.stride + 1
        output = np.zeros((self.num_filters, out_h, out_w))

        for f in range(self.num_filters):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * self.stride
                    h_end = h_start + self.filter_size
                    w_start = j * self.stride
                    w_end = w_start + self.filter_size
                    region = self.padded_input[:, h_start:h_end, w_start:w_end]
                    output[f, i, j] = np.sum(region * self.filters[f]) + self.biases[f]

        return output

    def backward(self, d_output, learning_rate):
        d_filters = np.zeros_like(self.filters)
        d_input = np.zeros_like(self.padded_input)
        _, out_h, out_w = d_output.shape

        for f in range(self.num_filters):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * self.stride
                    h_end = h_start + self.filter_size
                    w_start = j * self.stride
                    w_end = w_start + self.filter_size
                    region = self.padded_input[:, h_start:h_end, w_start:w_end]
                    d_filters[f] += d_output[f, i, j] * region
                    d_input[:, h_start:h_end, w_start:w_end] += (
                        d_output[f, i, j] * self.filters[f]
                    )

        self.filters -= learning_rate * d_filters
        self.biases -= learning_rate * np.sum(d_output, axis=(1, 2))

        if self.padding > 0:
            d_input = d_input[:, self.padding:-self.padding, self.padding:-self.padding]

        return d_input


class MaxPoolLayer:
    """Max pooling layer with forward and backward pass."""

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input_data):
        self.input_data = input_data
        num_filters, h, w = input_data.shape
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1
        output = np.zeros((num_filters, out_h, out_w))
        self.max_indices = np.zeros((num_filters, out_h, out_w, 2), dtype=int)

        for f in range(num_filters):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * self.stride
                    h_end = h_start + self.pool_size
                    w_start = j * self.stride
                    w_end = w_start + self.pool_size
                    region = input_data[f, h_start:h_end, w_start:w_end]
                    output[f, i, j] = np.max(region)
                    max_idx = np.unravel_index(np.argmax(region), region.shape)
                    self.max_indices[f, i, j] = [
                        h_start + max_idx[0],
                        w_start + max_idx[1],
                    ]

        return output

    def backward(self, d_output, learning_rate):
        d_input = np.zeros_like(self.input_data)
        _, out_h, out_w = d_output.shape

        for f in range(self.input_data.shape[0]):
            for i in range(out_h):
                for j in range(out_w):
                    max_h, max_w = self.max_indices[f, i, j]
                    d_input[f, max_h, max_w] += d_output[f, i, j]

        return d_input


class ReLU:
    """ReLU activation function."""

    def forward(self, input_data):
        self.input_data = input_data
        return np.maximum(0, input_data)

    def backward(self, d_output, learning_rate):
        return d_output * (self.input_data > 0)


class FlattenLayer:
    """Flatten multi-dimensional input into 1D vector."""

    def forward(self, input_data):
        self.input_shape = input_data.shape
        return input_data.flatten()

    def backward(self, d_output, learning_rate):
        return d_output.reshape(self.input_shape)


class FullyConnectedLayer:
    """Fully connected (dense) layer with forward and backward pass."""

    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(
            2.0 / input_size
        )
        self.biases = np.zeros(output_size)

    def forward(self, input_data):
        self.input_data = input_data
        return np.dot(input_data, self.weights) + self.biases

    def backward(self, d_output, learning_rate):
        d_input = np.dot(d_output, self.weights.T)
        d_weights = np.outer(self.input_data, d_output)
        d_biases = d_output

        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_input


class Softmax:
    """Softmax activation with cross-entropy loss."""

    def forward(self, input_data):
        exp_values = np.exp(input_data - np.max(input_data))
        self.output = exp_values / np.sum(exp_values)
        return self.output

    def backward(self, true_label, learning_rate):
        gradient = self.output.copy()
        gradient[true_label] -= 1
        return gradient

    def loss(self, true_label):
        return -np.log(self.output[true_label] + 1e-10)


class CNN:
    """
    Complete CNN architecture:
    Input (1x28x28) ->
    Conv(8 filters, 3x3) -> ReLU -> MaxPool(2x2) -> output: 8x13x13
    Conv(16 filters, 3x3) -> ReLU -> MaxPool(2x2) -> output: 16x5x5
    Flatten (400) -> FC(128) -> ReLU -> FC(10) -> Softmax
    """

    def __init__(self):
        self.conv1 = ConvolutionLayer(num_filters=8, filter_size=3, in_channels=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPoolLayer(pool_size=2, stride=2)

        self.conv2 = ConvolutionLayer(num_filters=16, filter_size=3, in_channels=8)
        self.relu2 = ReLU()
        self.pool2 = MaxPoolLayer(pool_size=2, stride=2)

        self.flatten = FlattenLayer()

        self.fc1 = FullyConnectedLayer(16 * 5 * 5, 128)
        self.relu3 = ReLU()
        self.fc2 = FullyConnectedLayer(128, 10)
        self.softmax = Softmax()

    def forward(self, image):
        out = self.conv1.forward(image)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)

        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)

        out = self.flatten.forward(out)

        out = self.fc1.forward(out)
        out = self.relu3.forward(out)
        out = self.fc2.forward(out)
        out = self.softmax.forward(out)

        return out

    def backward(self, true_label, learning_rate):
        gradient = self.softmax.backward(true_label, learning_rate)

        gradient = self.fc2.backward(gradient, learning_rate)
        gradient = self.relu3.backward(gradient, learning_rate)
        gradient = self.fc1.backward(gradient, learning_rate)

        gradient = self.flatten.backward(gradient, learning_rate)

        gradient = self.pool2.backward(gradient, learning_rate)
        gradient = self.relu2.backward(gradient, learning_rate)
        gradient = self.conv2.backward(gradient, learning_rate)

        gradient = self.pool1.backward(gradient, learning_rate)
        gradient = self.relu1.backward(gradient, learning_rate)
        gradient = self.conv1.backward(gradient, learning_rate)

    def train_step(self, image, label, learning_rate):
        output = self.forward(image)
        loss = self.softmax.loss(label)
        prediction = np.argmax(output)
        self.backward(label, learning_rate)
        return loss, prediction == label

    def predict(self, image):
        output = self.forward(image)
        return np.argmax(output)
