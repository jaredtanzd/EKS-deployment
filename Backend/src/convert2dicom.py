
## To convert nii files in PhysioNet ICH dataset to Dicom
#Source : https://stackoverflow.com/questions/14350675/create-pydicom-file-from-numpy-array


import os
import pydicom
import nibabel as nib
import numpy as np
import pydicom._storage_sopclass_uids

def convertNifti2dicom( nft_file , file_dir, index, seriesUID, studyInstanceUID, dataset_name='PhysioNet', orient='RAS'):

    '''
    ntf_file : Nifti object loaded using nibabel
    file_dir : Directory to store dicom 
    index : Index of the series as filename for dicom instance
    seriesUID : UID for the series, as this function only process one slice
    studyInstanceUID : UID for the study, as this function only process one slice
    orient : Orientation that Nifti is using, usually 
    '''
    #dicom_file = pydicom.dcmread('temp.dcm')
    meta = pydicom.Dataset()
    meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian  

    dicom_file = pydicom.dataset.Dataset()
    dicom_file.file_meta = meta

    dicom_file.is_little_endian = True
    dicom_file.is_implicit_VR = False

    dicom_file.Modality = "CT"
    dicom_file.SeriesInstanceUID = seriesUID
    dicom_file.StudyInstanceUID = studyInstanceUID
    dicom_file.FrameOfReferenceUID = pydicom.uid.generate_uid()
    dicom_file.PatientID = f'{dataset_name}-{file_dir.split("/")[1]}'
    dicom_file.PatientName = f'{dataset_name}-{file_dir.split("/")[1]}'
    dicom_file.SOPInstanceUID = pydicom.uid.generate_uid()

 
    affine = np.concatenate([nft_file.header['srow_x'][np.newaxis,:],
                             nft_file.header['srow_y'][np.newaxis,:],
                             nft_file.header['srow_z'][np.newaxis,:],
                             ])
    dicom_file.ImagePositionPatient = affine[:,-1].tolist()


    # Reference https://discovery.ucl.ac.uk/id/eprint/10146893/1/geometry_medim.pdf
    pixelSpacing = list(nft_file.header.get_zooms()) # for ex (0.42, 0.42 , 5) last dim is slice thickness 
    dicom_file.PixelSpacing = pixelSpacing[:2]

    a_lph = affine[:,1] / pixelSpacing[1] / np.array([-1,-1,1])
    b_lph = affine[:,0] / pixelSpacing[0] / np.array([-1,-1,1])

    dicom_file.ImageOrientationPatient = np.concatenate([a_lph , b_lph]).tolist()
    #dicom_file.ImageOrientationPatient = [1, 0, 0 , 0 , 1 ,0]
    dicom_file.ImageType = r"ORIGINAL\PRIMARY\AXIAL"

    
    #dicom_file = clear_header(dicom_file)
    data = np.array(nft_file.get_fdata())
    arr = data[:,:,index].astype('int16')
    #arr = arr.astype('uint16')
    dicom_file.Rows = arr.shape[0]
    dicom_file.Columns = arr.shape[1]
    dicom_file.PhotometricInterpretation = "MONOCHROME2"
    dicom_file.SamplesPerPixel = 1
    dicom_file.BitsStored = 16
    dicom_file.BitsAllocated = 16
    dicom_file.HighBit = 15
    dicom_file.PixelRepresentation = 1
    #dicom_file.RescaleIntercept = -1024
    dicom_file.RescaleIntercept = 0
    dicom_file.RescaleSlope = 1
    dicom_file.RescaleType = 'HU'
    dicom_file.SliceThickness = 5.0 # for physionet
    dicom_file.StudyTime = ''
    dicom_file.StudyDate = ''
    dicom_file.InstanceNumber = index
    dicom_file.WindowCenter = 30
    dicom_file.WindowWidth = 100

    pydicom.dataset.validate_file_meta(dicom_file.file_meta, enforce_standard=True)
    
    if orient == 'RAS':
        #dicom_file.PixelData = arr[::-1,::-1].tobytes()  #Physionet arr[:,::-1].T 
        dicom_file.PixelData = arr[:,:].T.tobytes()  #Physionet arr[:,::-1].T 
    else:
        raise NotImplementedError
   
    try:
        dicom_file.save_as(os.path.join(file_dir, f'{index}.dcm'), write_like_original=False)
    except Exception as e:
        print(e)


def clear_header(dcm_file):
    attr = [f for f in dir(dcm_file) if ('_' not in f)&(f[0] == f[0].upper()) ]
    for a in attr:
        setattr(dcm_file, a, '')

    return dcm_file



if __name__ == '__main__':
    for n in os.listdir('ct_scans'):
        assert '.nii' in n, 'File should be Nifti format.'
        data = nib.load('ct_scans/'+ n) #try without filling header and orientation 
        name = n.split('.nii')[0]
        print(f'Processing {name} ...')
        seriesUID = pydicom.uid.generate_uid()
        studyInstanceUID = pydicom.uid.generate_uid()
        for slice_ in range(data.get_fdata().shape[-1]):
            dir_ = f'ct_scans_dcm/{name}'
            os.makedirs(dir_,exist_ok=True)
            convertNifti2dicom(data,dir_, slice_ ,seriesUID, studyInstanceUID)
     
     

