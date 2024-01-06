# hopfield-mnist

## Usage

```bash
# See mnist data by using matplotlib:
$ poetry run python mnist.py
# Generate the prediction data under ./png directory by using hopfield network:
$ poetry run python hopfield.py
```

## Sample Result

### Denoise (synchronous update)
<img src=./png/sync_mnist_image_prediction.png width=400px>

### Energy transition (synchronous update)
<img src=png/sync_energy_transition.png width=400px>

### Denoise (asynchronous update)
<img src=png/async_mnist_image_prediction.png width=400px>

### Energy transition (asynchronous update)
<img src=png/async_energy_transition.png width=400px>
