from re import sub
import monai
import os
import torch
import numpy as np
import nibabel as nib


def getSubvolumeOrigin(img, subvolume_size): # HACK FUNCTION FOR TEMPORARY FIX OF MEMORY ISSUE.
        """Get the origin of the cubic subvolume at the centre of the image. This is just a temporary solution to the memory issue."""
        xlen = img.shape[0]
        ylen = img.shape[1]
        zlen = img.shape[2]
        xmin = max(int((xlen-subvolume_size[0])/2),0)
        ymin = max(int((ylen-subvolume_size[1])/2),0)
        zmin = max(int((zlen-subvolume_size[2])/2),0) #hack for v02 images
        return (xmin, ymin, zmin)

def one_hot(label_batch, num_labels=87):
    """One-hot encode labels in a batch of label images.
    input shape : (batch_size, 1, H, W, D)
    output shape: (batch_size, num_labels+1, H, W, D)"""

    num_classes = num_labels + 1 # Treat background (label 0) as a separate class).
    label_batch = monai.networks.utils.one_hot(label_batch, num_classes, dim=1) # one-hot encode the labels    
    return label_batch

seed=489
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = monai.networks.nets.HighResNet(spatial_dims=3,
                                        in_channels=1,
                                        out_channels=88,
                                        dropout_prob=0.0
)
        
model.to(device)

home_dir = '/hpf/largeprojects/smiller/users/Katharine/flask_project/predict'
model_dir = os.path.join(home_dir, 'models')

#template_path = os.path.join(home_dir,'templates/sub-CC00060XX03_ses-12501_desc-drawem87_space-T2w_dseg.nii.gz')
#template_path = os.path.join(home_dir,'templates/BC0001_V01.nii.gz')
# image_dir = os.path.join(home_dir,'/image')
out_dir = os.path.join(home_dir,'../predicted_images')
temp_file_name = os.path.join(home_dir,'image_path.txt')
temp_file = open(temp_file_name,"r")
image_dir = os.path.join(home_dir,'../uploaded_images')
image_name = temp_file.readline().strip()
model_name = temp_file.readline().strip()



print("IMAGE", image_name)
print("MODEL", model_name)
temp_file.close()
model_path = os.path.join(model_dir, model_name)
best_state_dict = torch.load(model_path)
# out_dir = '/hpf/largeprojects/smiller/users/Katharine/brain_segmentation/segment/predict/out'
max_subvolume_size = [150,150,150]

#model.to(device)

model.load_state_dict(best_state_dict)
model.eval()

image_path = os.path.join(image_dir, image_name)
template_path = image_path
img = np.array(nib.load(image_path).get_fdata(), dtype=np.float32) # load NIFTI file into numpy array.
print(img.shape)
# xmin, ymin, zmin = getSubvolumeOrigin(img, max_subvolume_size)


subvolume_size = [min(max_subvolume_size[0],img.shape[0]),min(max_subvolume_size[1],img.shape[1]),min(max_subvolume_size[2],img.shape[2])]
xmin, ymin, zmin = getSubvolumeOrigin(img, subvolume_size)
subvolume_size = [min(xmin+subvolume_size[0], img.shape[0]), min(ymin+subvolume_size[1], img.shape[1]), min(zmin+subvolume_size[2], img.shape[2])]

img = img[xmin:subvolume_size[0], ymin:subvolume_size[1], zmin:subvolume_size[2]]
print(img.shape)
Normalizer = monai.transforms.NormalizeIntensity()
img = Normalizer(img)

img = np.expand_dims(np.expand_dims(img, axis=0),axis=0)
#img = np.expand_dims(img, axis=0)



img = torch.tensor(img)
img = img.to(device)

yhat = model(img) # shape (B=1, num_labels+1, H, W, D)

## We want to save the predicted labels as a NIFTI file.
yhat = yhat.to('cpu')
yhat = yhat.detach().numpy()
yhat = np.array(yhat)
yhat = yhat[0, :, :, :, :] # shape (num_labels+1, H, W, D) # get first item in batch (batch size is 1 here)
yhat = np.argmax(yhat, axis=0) # shape (H, W, D) # Convert from one-hot-encoding back to labels.
labels_orig = nib.load(template_path) # Load the original labels NIFTI file to use as a template.
#label_pred_data = np.zeros(shape=yhat.shape) # Initialize the predicted labels image as an array of zeros of the same size as original labels image. 
label_pred_data = np.zeros(shape=labels_orig.get_fdata().shape)
#print(xmin,ymin,zmin)

#### HACK: Fill with predicted values in subvolume cube in which prediction were made.
#xmin, ymin, zmin = getSubvolumeOrigin(img)
#label_pred_data = yhat
label_pred_data[xmin:subvolume_size[0], ymin:subvolume_size[1], zmin:subvolume_size[2]] = yhat

#label_pred_data[xmin:subvolume_size[0], ymin:subvolume_size[1], zmin:subvolume_size[2]] = yhat
label_pred_nifti = nib.Nifti1Image(label_pred_data, labels_orig.affine, labels_orig.header) # Construct a NIFTI file using the predicted label data, but the header and affine matrix from the original labels NIFTI file.

model_name_short = model_name.split('.')[0]
# Save predicted labels image as a NIFTI file.
out_name = model_name_short+'_'+image_name
out_path = os.path.join(out_dir, out_name)
nib.save(label_pred_nifti, out_path)

temp_file_name = os.path.join(home_dir,'predicted_path.txt')
temp_file = open(temp_file_name,"w")
temp_file.write(out_name)
temp_file.close()

print("IMAGE READY")