import numpy as np
import gzip
import os
import urllib.request
from cnn import CNN


FASHION_MNIST_URLS = {
    "train_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
    "train_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
    "test_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
    "test_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
}

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def download_fashion_mnist(data_dir="data"):
    """Download Fashion MNIST dataset if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    files = {}
    for key, url in FASHION_MNIST_URLS.items():
        filename = os.path.join(data_dir, url.split("/")[-1])
        if not os.path.exists(filename):
            print(f"Downloading {key}...")
            urllib.request.urlretrieve(url, filename)
        files[key] = filename
    return files


def load_images(filepath):
    """Load images from gzipped IDX file."""
    with gzip.open(filepath, "rb") as f:
        _ = int.from_bytes(f.read(4), "big")
        num_images = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_images, rows, cols)


def load_labels(filepath):
    """Load labels from gzipped IDX file."""
    with gzip.open(filepath, "rb") as f:
        _ = int.from_bytes(f.read(4), "big")
        _ = int.from_bytes(f.read(4), "big")
        return np.frombuffer(f.read(), dtype=np.uint8)


def preprocess(images):
    """Normalize pixel values to [0, 1]."""
    return images.astype(np.float64) / 255.0


def train(model, train_images, train_labels, epochs=3, learning_rate=0.005, num_samples=1000):
    """
    Train the CNN model on Fashion MNIST.

    Args:
        model: CNN instance
        train_images: normalized training images
        train_labels: training labels
        epochs: number of training epochs
        learning_rate: step size for gradient descent
        num_samples: number of samples per epoch (for feasibility with pure numpy)
    """
    print(f"\n{'='*60}")
    print(f"Training CNN on Fashion MNIST")
    print(f"Epochs: {epochs}, Learning Rate: {learning_rate}, Samples/Epoch: {num_samples}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        indices = np.random.permutation(len(train_images))[:num_samples]
        total_loss = 0
        correct = 0

        for idx, i in enumerate(indices):
            image = train_images[i]
            label = train_labels[i]

            loss, is_correct = model.train_step(image, label, learning_rate)
            total_loss += loss
            correct += int(is_correct)

            if (idx + 1) % 100 == 0:
                avg_loss = total_loss / (idx + 1)
                accuracy = correct / (idx + 1) * 100
                print(
                    f"  Epoch {epoch+1}/{epochs} | "
                    f"Step {idx+1}/{num_samples} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Accuracy: {accuracy:.1f}%"
                )

        epoch_loss = total_loss / num_samples
        epoch_acc = correct / num_samples * 100
        print(
            f"\n  Epoch {epoch+1} Complete - "
            f"Avg Loss: {epoch_loss:.4f}, "
            f"Accuracy: {epoch_acc:.1f}%\n"
        )

    return model


def evaluate(model, test_images, test_labels, num_samples=500):
    """
    Evaluate the CNN model on test data.

    Args:
        model: trained CNN instance
        test_images: normalized test images
        test_labels: test labels
        num_samples: number of test samples to evaluate
    """
    print(f"\n{'='*60}")
    print(f"Evaluating CNN on {num_samples} test samples")
    print(f"{'='*60}\n")

    indices = np.random.permutation(len(test_images))[:num_samples]
    correct = 0
    class_correct = np.zeros(10)
    class_total = np.zeros(10)

    for idx, i in enumerate(indices):
        image = test_images[i]
        label = test_labels[i]

        prediction = model.predict(image)
        is_correct = prediction == label
        correct += int(is_correct)
        class_total[label] += 1
        class_correct[label] += int(is_correct)

        if (idx + 1) % 100 == 0:
            print(f"  Evaluated {idx+1}/{num_samples} samples...")

    overall_accuracy = correct / num_samples * 100
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"\n  Overall Accuracy: {overall_accuracy:.2f}% ({correct}/{num_samples})\n")
    print(f"  Per-class Accuracy:")
    print(f"  {'-'*40}")

    for c in range(10):
        if class_total[c] > 0:
            acc = class_correct[c] / class_total[c] * 100
            print(f"    {CLASS_NAMES[c]:15s}: {acc:5.1f}% ({int(class_correct[c])}/{int(class_total[c])})")

    print(f"\n{'='*60}\n")
    return overall_accuracy


def main():
    print("Fashion MNIST CNN - Built from Scratch using NumPy")
    print("=" * 60)

    files = download_fashion_mnist()

    print("\nLoading dataset...")
    train_images = preprocess(load_images(files["train_images"]))
    train_labels = load_labels(files["train_labels"])
    test_images = preprocess(load_images(files["test_images"]))
    test_labels = load_labels(files["test_labels"])

    print(f"  Training set: {train_images.shape[0]} images")
    print(f"  Test set: {test_images.shape[0]} images")
    print(f"  Image size: {train_images.shape[1]}x{train_images.shape[2]}")
    print(f"  Classes: {len(CLASS_NAMES)}")

    model = CNN()

    model = train(
        model,
        train_images,
        train_labels,
        epochs=3,
        learning_rate=0.005,
        num_samples=1000,
    )

    evaluate(model, test_images, test_labels, num_samples=500)


if __name__ == "__main__":
    main()
