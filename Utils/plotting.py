import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

def generate_tensors_from_files(image_file, mask_file):

    input_image = Image.open('lgg-mri-segmentation/kaggle_3m/TCGA_DU_7018_19911220/TCGA_DU_7018_19911220_21.tif').convert('RGB')
    ground_truth_mask = Image.open('lgg-mri-segmentation/kaggle_3m/TCGA_DU_7018_19911220/TCGA_DU_7018_19911220_21_mask.tif').convert('L')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    # Preprocess the input image
    input_tensor = transform(input_image)
    ground_truth_tensor = transform(ground_truth_mask)

    return input_tensor, ground_truth_tensor

def generate_images(input_img_tensor, input_mask_tensor, model):

    '''
    Return 3 images for plotting
    Takes in squeezed tensors
    Outputs are ready for plotting with matplotlib
    '''

    input_tensor = input_img_tensor.unsqueeze(0)
    # Perform inference

    model.to('cpu')

    with torch.no_grad():
        model.eval()
        output_tensor = model(input_tensor)

    # Convert the output tensor to a numpy array
    predicted_mask = output_tensor.squeeze(0).cpu().numpy()
    print(predicted_mask.max())

    mask_img = predicted_mask[0]

    for i in range(len(mask_img)):
        for j in range(len(mask_img[i])):
            if mask_img[i][j] < 0:
                predicted_mask[0][i][j] = -13
            else:
                predicted_mask[0][i][j] = 13

    # Display input image, ground truth mask, and predicted mask
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    input_image =np.transpose(input_img_tensor, (1, 2, 0))
    ground_truth_mask = np.transpose(input_mask_tensor, (1, 2, 0))

    return input_image, ground_truth_mask, predicted_mask