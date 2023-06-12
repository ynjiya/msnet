import pathlib
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from config import *
from dataset.image_utils import pad_or_crop_image, irm_min_max_preprocess, zscore_normalise
from monai.transforms import apply_transform


class MSD(Dataset):
    def __init__(self, arr, on="train", data_aug=False, normalisation="minmax", transform=None):
        super(MSD, self).__init__()
        self.data_aug = data_aug
        self.normalisation = normalisation
        self.on = on
        self.transform = transform
        
        self.datas = []

        if on == "eval":
            for i in arr:
                self.datas.append(dict(id=int(i[0]), img=i[1]))
        else:
            for i in arr:
                self.datas.append(dict(id=int(i[0]), img=i[1], seg=i[2]))


    def _transform(self, index: int):
        data_i = self.data[index]
        return apply_transform(self.transform, data_i) if self.transform is not None else data_i


    def __getitem__(self, idx):
        _patient = self.datas[idx]
        patient_image = self.load_nii(_patient["img"])
        size_before_crop = patient_image.shape

        # crop to non-zero 
        # remove maximum extent of the zero-background to make future crop more useful
        z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
        # add 1 pixel at each side
        zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
        zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
        patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]

        # intensity normalization
        if self.normalisation == "minmax":
            patient_image = irm_min_max_preprocess(patient_image)
        elif self.normalisation == "zscore":
            patient_image = zscore_normalise(patient_image) 

        if self.on != "eval":
            patient_label = self.load_nii(_patient["seg"])
            et = patient_label == 3
            et_present = 1 if np.sum(et) >= 1 else 0
            tc = np.logical_or(et, patient_label == 2)
            wt = np.logical_or(tc, patient_label == 1) 
            patient_label = np.stack([et, tc, wt])
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]

        # crop to (128, 128, 128)
        if self.on == "train" or self.on == "val":
            patient_image, patient_label = pad_or_crop_image(patient_image, patient_label, target_size=(128, 128, 128))

        patient_image = torch.from_numpy(patient_image.astype("float16"))
        if self.on == "eval":
            return dict(
                        id=_patient["id"],
                        image=patient_image,
                        og_size=size_before_crop,
                        crop_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
                        supervised=True,
                        )

        patient_label = torch.from_numpy(patient_label.astype("bool"))
        return dict(
                    id=_patient["id"],
                    image=patient_image,
                    label=patient_label,
                    seg_path=str(_patient["seg"]),
                    crop_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
                    et_present=et_present,
                    supervised=True,
                    )

    @staticmethod
    def load_nii(path_folder):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))

    def __len__(self):
        return len(self.datas)


def get_datasets(on="train", normalisation="minmax", 
                 evalpath=EVAL_DATA_FOLDER_IN, 
                 mainpath=MAIN_DATA_FOLDER_MSD):
    if on == "eval":
        eval_folder = pathlib.Path(evalpath).resolve()
        imagesTs = [x for x in eval_folder.glob('*.nii.gz')]
        eval_dataset = [[str(imagesTs[i])[-10:-7] , imagesTs[i]] for i in range(len(imagesTs))]
        return MSD(eval_dataset, on="eval", normalisation=normalisation)

    base_folder = pathlib.Path(mainpath, "imagesTr").resolve()
    assert base_folder.exists()
    label_folder = pathlib.Path(mainpath, "labelsTr").resolve()
    assert label_folder.exists()

    imagesTr = [x for x in base_folder.glob('*.nii.gz')]
    labelsTr = [x for x in label_folder.glob('*.nii.gz')]
    dataset = [[str(imagesTr[i])[-10:-7] , imagesTr[i], labelsTr[i]] for i in range(len(imagesTr))]
    
    # split dataset 80:15:5
    train, val, test = np.array_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.95)])

    test_dataset = MSD(test, on="test", normalisation=normalisation)
    if on == "test":
        return test_dataset

    train_dataset = MSD(train, on="train", normalisation=normalisation)
    val_dataset = MSD(val, on="val", data_aug=False, normalisation=normalisation)
    return train_dataset, val_dataset, test_dataset