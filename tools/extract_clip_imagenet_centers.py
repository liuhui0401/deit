import torch
import os
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, AutoProcessor
from torchvision import transforms
import functools
import numpy as np
import h5py
import json

def save_to_hdf5(features, labels, filename):
    assert len(features) == len(labels), "Number of features and labels must be the same."

    with h5py.File(filename, "w") as f:
        # Create datasets to store features and labels
        features_dset = f.create_dataset("features", data=features)
        labels_dset = f.create_dataset("labels", data=[label.encode("utf-8") for label in labels])


def merge_hdf5(input_files, output_file):
    with h5py.File(output_file, "w") as out_f:
        for i, input_file in enumerate(input_files):
            with h5py.File(input_file, "r") as in_f:
                # Iterate over datasets in input file
                for dataset_name, dataset in in_f.items():
                    if dataset_name not in out_f:
                        # Create dataset in output file
                        out_dset = out_f.create_dataset(dataset_name, data=dataset, chunks=True, maxshape=(None,)+dataset.shape[1:])
                    else:
                        # Append dataset to existing dataset in output file
                        out_dset = out_f[dataset_name]
                        out_dset.resize((out_dset.shape[0] + dataset.shape[0]), axis=0)
                        out_dset[-dataset.shape[0]:] = dataset[:]


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

# Define a custom dataset class
class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, preprocess):
        self.folder_path = folder_path
        self.preprocess = preprocess
        self.image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if
                            filename.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self.preprocess(Image.open(image_path).convert('RGB'))
        return image, image_path

# Function to extract image features using DataLoader
def extract_image_features_dataloader(data_loader, model):
    image_features = []

    for images, paths in tqdm(data_loader, desc="Extracting image features using DataLoader"):
        images = images.to(device)

        with torch.no_grad():
            batch_features = model(images)
            batch_features = batch_features.image_embeds

        image_features.extend(batch_features.cpu().numpy())

    return image_features


if __name__ == '__main__':
    # Load the CLIP model
    device = "cuda"
    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").cuda()
    preprocess = transforms.Compose([
        transforms.Lambda(
            functools.partial(center_crop_arr, image_size=224)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466,0.4578275,0.40821073], std=[0.26862954,0.26130258,0.27577711],
                             inplace=True),
    ])
    model = model.to(device)

    center_dict = {}
    root = '/data0/data/imagenet/train/'
    os.makedirs('h5_features', exist_ok=True)
    for ff in os.listdir(root):
        if not ff.startswith('n0'):
            continue
        # Set the folder path
        folder_path = os.path.join(root, ff)

        # Create an instance of the custom dataset
        dataset = ImageFolderDataset(folder_path, preprocess)

        # Define batch size
        batch_size = 250

        # Create a DataLoader
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        # Extract image features using DataLoader
        image_features = extract_image_features_dataloader(data_loader, model)

        # Convert gathered image features to numpy arrays
        all_image_features_numpy = np.array(image_features)
        print(f'all_image_features_numpy shape: {all_image_features_numpy.shape}')

        center_dict[ff] = all_image_features_numpy.mean(0).tolist()

        save_to_hdf5(all_image_features_numpy, [ff] * all_image_features_numpy.shape[0], f'h5_features/{ff}.h5')

        # import ipdb
        # ipdb.set_trace()

        with open('centers_feature.json', 'w') as f:
            f.write(json.dumps(center_dict))