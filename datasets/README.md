# dataset

We used here [Cityscapes dataset](https://www.cityscapes-dataset.com/).
You can download it directly from Kaggle using the following code.
This Python snippet downloads the whole dataset used to train pix2pix.

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("vikramtiwari/pix2pix-dataset")
print("Path to dataset files:", path)
```

We are using only the `cityscapes` for testing. For this you need to use the dataset class in `cityscapes_dataset.py`.

