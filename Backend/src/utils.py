import os
import socket
import multiprocessing
from pathlib import Path
from abc import abstractmethod
from collections import OrderedDict, defaultdict
import json
from typing import Dict, List, Tuple
from collections import defaultdict 
import glob
import numpy as np
import pandas as pd
import scipy
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from skimage import morphology
import seaborn as sns
import pydicom
from pydicom.dataset import FileDataset
import datetime
import yaml
from src.fetch import *
from src.awsclient import Bucket, Dummy
#from src.models.ICH_yolo.segmentation.em import EM_segm
#from src.model import unet
#import torchio as tio


def gethost():
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    return hostname , ip


#### TODO CURRENT SOLUTION 
PROJECT = ['ich','tissueseg']
SERVER_IP = gethost()[1]

if 'bionet' in gethost()[0]:
    CACHE = '/home/data_repo/cache/'
    bucket = Dummy()

else:
    SERVER_IP = 'XX'
    CACHE = '/tmp'
    bucket = Bucket(CACHE)

os.makedirs(CACHE, exist_ok=True)




def get_current_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', 0))
    port = sock.getsockname()[1]
    port = 5000
    return port


class DicomTensor:
    '''
    Light weight object to store DicomTags

    '''

    def __init__(self, dcm):
        assert isinstance(dcm , FileDataset) , 'Object is not Dicom FileDataset.'
        self.__get_attr(dcm)

    def __get_attr(self,dcm):
        dic = {}
        for att in dir(dcm):
            if att == 'PixelData': continue 
            if '__' in att: break #all attr before hidden ones , does not include pixel_array
            dic[att] = getattr(dcm , att)
         
        self.__dict__ = dic

    def __repr__(self):
        return self.SOPInstanceUID

    def add(self, im):

        self.im = im

    def attr(self, key, value):
        self.__dict__[key] = value

    @property
    def vis(self):
        img = self.im
        img = img.astype(np.float64)
        img = (255*(img - np.min(img))/ (np.ptp(img) + 1e-6)).astype(np.float64)
        img = img.astype(np.uint8)
        return img


class Volume:
    def __init__(self, project='ich'):
        self.df = []
        self.proj = project.lower()
        

    def update(self,kw : Tuple, timestamp ):
        '''
        kw : (dcmobj, target label , Number of pixels)
        '''
        assert len(kw) == 3, 'Only Tuple or List of length 2 is accepted. Ex: ("ID" ,"EPH", 100)'
        assert isinstance(kw[1], str)
        dcmobj = kw[0]
        self.thickness = dcmobj.SliceThickness
        self.spacing = dcmobj.PixelSpacing
        assert len(self.spacing) == 2, 'Dicom does not have length 2 pixel spacing.'
        self.voxelsize = self.thickness * self.spacing[0] * self.spacing[1]

        dic = {}
        dic['series_id'] = kw[0].orthancSeriesID
        dic['instance_id'] = kw[0].orthancID
        dic['target'] = kw[1]
        dic['volume'] = kw[2] * self.voxelsize
        dic['timestamp'] = timestamp
        self.df += [dic]

    def flush(self):
        path = Path(CACHE) / f'{self.proj}/volume.csv'
        existing = pd.read_csv(path)
        df = pd.DataFrame(self.df)
        pd.concat([df, existing]).to_csv(path, index=False)
        hdr = False  if os.path.isfile(path) else True
        #df.to_csv(path, mode='a', index=False, header=hdr)


def convert_png_transparent(filename, default_color=(255, 163, 163)):
    """
    
    This function assumes that the top left pixel = background 
    
    Input: filename of .png file with RGB channels
    """
    img_grey = Image.open(filename).convert('L')
    img_grey_data = np.array(img_grey) # 1 channel only
    assert len(np.unique(img_grey_data)) <= 2 # greyscale img should only have 2 values: mask and background
    background_value = img_grey_data[0][0] # assume top left pixel is bg
    if background_value >0:
        img_grey_data[np.where(img_grey_data==background_value)] = 0
    
    alpha = np.zeros(np.shape(img_grey_data))
    alpha[np.where(img_grey_data != 0)] = 255//2 # set alpha for mask areas
    # add alpha channel to png
    output = img_grey.convert("RGB")
    output_data = np.array(output)
    #output_data[np.where(img_grey_data != 0)] = [255, 163, 163] 
    output_data[np.where(img_grey_data != 0)] = default_color #129,216,208
    output_data = np.dstack((output_data, alpha)).astype('uint8')#, axis=-1)
    output = Image.fromarray(output_data)
    output.save(filename)


#def to_json(pred: Dict , mask: Dict, x : DicomTensor): 
#
#    '''
#    Convert model output to json
#
#    Args:
#      pred: {uid1: [ [x1,y1,x2,y2, class] , [x1,y1,x2,y2 , class] ] }
#      mask: {uid1: [ numpy.array [512 x 512] , ... ]}
#
#
#    Output:
#      { InstanceID:
#                      { Subtype1: { url: str , bbox: List} },... }
#    '''
#
#    label_names= ['chronic','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']
#    main_json = {}
#    complete_mask = {}
#    is_series = len(x) > 1
#    vol = Volume()
#    project= 'ich'
#    curr_time = datetime.datetime.now()
#    for uid in pred.keys(): #uid is the ID in dicom image
#        dcmobj = [t for t in x if t.SOPInstanceUID == uid][0]
#        save_uid = dcmobj.orthancID #save_uid is the ID used in dicom server - frontend - backend
#        main_json[save_uid] = {}
#        
#        cache_path = str(Path(CACHE) / project / save_uid)
#
#        if not os.path.exists(Path(cache_path)): #check dir
#            os.makedirs(Path(cache_path))
#
#        ##====================================
#        #get preprocessed image here
#        x_path = str(Path( cache_path))+  '_inter.png'
#        img = Image.fromarray(dcmobj.vis.astype(np.uint8))
#        img.save(x_path, cmap='gray')
#        ##====================================
#
#        ## declare a master mask
#        M = np.zeros_like(img).astype(np.int)
#        M_bb = np.zeros_like(img).astype(np.uint8)
#        for i, label in enumerate(label_names): #look for subtype
#            subtype_bbox_mask = [(arr,mask[uid][j] ) for j,arr in enumerate(pred[uid]) if arr[-1]==i] #slice bbox of the subtype, and corresponding mask
#            subtype_dict = {}
#            for k, (bb, m) in enumerate(subtype_bbox_mask): #Subtype1-(k)
#            ## adding the mask
#                M += m
#
#                # bbox mask
#                x1,y1,x2,y2 = bb[:4].astype(int)
#                M_bb = cv2.rectangle(np.array(M_bb), (x1, y1), (x2, y2), (255,0,0), 2)
#                #cv2.putText(M_bb, label, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 0.7, (36,255,12), 2)
#                cv2.putText(M_bb, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
#
#                subtype_dict[k] = {}
#                # save im at url , path
#                subtype_dict[k]['url'] = os.path.join(cache_path, f'{label}_{k}.png')
#                matplotlib.image.imsave(subtype_dict[k]['url'], m)
#                #bbox
#                subtype_dict[k]['bbox'] = [str(int(x)) for x in bb[:4]]
#                #confidence
#                subtype_dict[k]['conf'] = str(bb[-2])
#                
#                if is_series:
#                    vol.update((dcmobj, label, (m>0).sum()), curr_time)
#
#
#
#            main_json[save_uid][label] = subtype_dict
#
#        #whole mask
#        #if M.sum() > 0:
#            #matplotlib.image.imsave(cache_path+'.png' , (M>0).astype(int))
#        np.save(cache_path+'.npy',(M>0).astype(np.uint8) )
#            #convert_png_transparent(cache_path + '.png')
#        complete_mask[save_uid] = f'{SERVER_IP}:{get_current_port()}/retrieve/ich/?instance={save_uid}'
#
#            #save bbox mask
#        cv2.imwrite(cache_path + '_bb.png',M_bb )
#        convert_png_transparent(cache_path + '_bb.png')
#
#
#        with open(os.path.join(os.path.dirname(cache_path), 'meta.json'), 'w') as f:
#            json.dump(main_json, f, indent=4, sort_keys=False)
#
#
#    if is_series: 
#        vol.flush()
#        
#    return main_json, complete_mask
#


def DicomObj(func):
    storeval = True
    def Store(self, dcm, return_list=None):
        dcmtensor = DicomTensor(dcm)
        dcmtensor.add(func(self,dcm))
        if return_list is not None: 
            return_list += [dcmtensor]
        return dcmtensor

    def NonStore(self, dcm, args):
        return func(self,dcm)

    if storeval: return Store
    else: return NonStore
      

class Parallel():
    #TODO currently do not restrict number of cores, use all cores
    def __init__(self, n_node = multiprocessing.cpu_count()):

        self.processes = []
        self.manager = multiprocessing.Manager()
        self.n_node = min(n_node, multiprocessing.cpu_count())

    def __call__(self,func, input):
        
        return_list = self.manager.list()
        
        for i in range(len(input)):
            p = multiprocessing.Process(target=func, args=(input[i],return_list))
            self.processes.append(p)
            p.start()

        for process in self.processes:
            process.join()

        return return_list


class BasePipeline:

    '''
    A blueprint data preprocessing pipeline for all modalities.
    Child class should include run_pipeline and all the supporting functions.
    Check CT_Pipeline for reference.

    '''

    def __init__(self, storeval, parallel):
        self.default_shape = (512,512)
        self.storeval = storeval
        self.parallel = parallel
        if parallel:
            self.parallel_fn = Parallel()

    @abstractmethod
    def run_pipeline(self,dcm):
        return

    @DicomObj
    def run(self,dcm, return_list: List = None):
        '''
        return_list: A list for multiprocessing
        '''
        if return_list is None:
            return self.run_pipeline(dcm)

    
    def __enter__(self):
        return self

    def __exit__(self,exception_type, exception_value, traceback):
        pass


    def __call__(self, dicom : List):
        if not isinstance(dicom , list) : dicom = [dicom]      

        if isinstance(dicom, list):
            assert all(list(map(lambda x: isinstance(x, FileDataset) , dicom))) , 'Input must be dicom object.'

        if (len(dicom) > 10) & self.parallel:
            newlist = self.parallel_fn(self.run, dicom)
        else:
            newlist = list(map(lambda x: self.run(x), dicom ))

        #try:
        #    newlist = sorted(newlist, key=lambda x: x.SliceLocation)
        #except AttributeError as e:
        #    if len(newlist) == 1: pass
        #    else: raise e

        if not self.storeval:
            newlist = [obj.im for obj in newlist]

        return newlist


class CT_Pipeline(BasePipeline):

    def __init__(self, wl = 40, ww=80, storeval=True, parallel=True):
        '''
        storeval : To create an obj for each preprocessed array to retain its dicom tag.
        '''
        super().__init__(storeval, parallel)
       
        self.wl = wl
        self.ww = ww

    def winsorise(self,raw_dcm):

        if self.wl is None : self.wl = raw_dcm.WindowCenter if not isinstance(raw_dcm.WindowCenter,pydicom.multival.MultiValue) else 40
        if self.ww is None : self.ww = raw_dcm.WindowWidth if not isinstance(raw_dcm.WindowWidth,pydicom.multival.MultiValue) else 80

        raw_data = raw_dcm.pixel_array.copy()
        raw_data[raw_data == -2000] = 0
        return raw_data

    @staticmethod
    def rescale(raw_dcm, image_data):
        return (image_data * raw_dcm.RescaleSlope) + raw_dcm.RescaleIntercept

    @staticmethod
    def window(raw_dcm, window_center, window_width):
        img = raw_dcm.copy()
        img_min = window_center - window_width//2
        img_max = window_center + window_width//2
        img[img<img_min] = img_min
        img[img>img_max] = img_max
        return img 

    @staticmethod
    def normalise(img, window_center, window_width):
        img_min = window_center - window_width//2
        img_max = window_center + window_width//2
        img = (img - img_min) / (img_max - img_min)
        return img
    
    @staticmethod
    def generate_brain_mask(windowed):
    
        '''
        Seems to need brain window to work
        '''
       
        segmentation = morphology.dilation(windowed, np.ones((1, 1)))
        labels, label_nb = scipy.ndimage.label(segmentation)
    
        label_count = np.bincount(labels.ravel().astype(np.int))
        label_count[0] = 0
    
        mask = labels == label_count.argmax()
        mask = morphology.dilation(mask, np.ones((1, 1)))
        mask = scipy.ndimage.morphology.binary_fill_holes(mask)
        mask = morphology.dilation(mask, np.ones((3, 3)))

        return mask

    def run_pipeline(self,raw_dicom):
        out = self.winsorise(raw_dicom)
        out = CT_Pipeline.rescale(raw_dicom, out)
        out = CT_Pipeline.window(out, self.wl, self.ww)
        mask = CT_Pipeline.generate_brain_mask(out)
        #gantry tilt
        out = mask * out
        #if out.shape != self.default_shape:
        #    out = cv2.resize( out.astype(float), self.default_shape , cv2.INTER_AREA)
        # out = CT_Pipeline.normalise(out, self.wl, self.ww)

        return out

def interp_points(x1,y1,x2,y2):

    if x2 == x1:
        segment_points = (np.array([x1]*int(abs(y2-y1))), y1+np.arange(0,y2-y1,1 if y2>y1 else -1))
    elif x2 > x1:
        segment_points = (x1+np.arange(0, x2-x1+1,1) , np.interp(x1 + np.arange(0, x2-x1+1, 1) , (x1,x2), (y1,y2)))
    else:
        segment_points = (x2+np.arange(0, x1-x2 +1,1) , np.interp(x2 + np.arange(0, x1-x2 +1, 1) , (x2,x1), (y2,y1)))
    return segment_points


def rotation_calib(im_path ):
    #https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
    #https://stackoverflow.com/questions/28710337/opencv-lines-passing-through-centroid-of-contour-at-given-angles
    img = cv2.imread(im_path)
    im_path = str(Path(im_path.parent) / Path(im_path.stem)) + '_rot.png'
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray_image,250,255,0)
    M = cv2.moments(thresh)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    N = 180 // 2 #each delta is 2 deg
    width = img.shape[1]
    height = img.shape[0]
    segments = {}
    skull_thickness = []
    for i in range(N):
  
        # Step #6b
        theta = i*(180/N)
        theta2 = theta + 180
        theta *= np.pi/180.0
        theta2 = theta2 *np.pi/180.0
        segment_mask = np.zeros_like(thresh)
  
        # Step #6c
        #cv2.line(img, (cX, cY),
        #         (int(cX +np.cos(theta)*width),
        #          int(cY-np.sin(theta)*height)), 255, 1)

        #cv2.line(img,
        #         (int(cX +np.cos(theta2)*width),
        #          int(cY-np.sin(theta2)*height)), (cX, cY), 255, 1)
        #make each line an obj or so, mask the line with the thresh, the sum the intersected pixel
        #use interp to get the pixels on the cross lines, then  mask with skull


        #Added (derive the points in #6c cv2.line visualization)
        segment = interp_points(cX,cY, int(cX +np.cos(theta)*width), int(cY-np.sin(theta)*height)) 

        segment_poly = np.concatenate([segment[0][:,np.newaxis], segment[1][:,np.newaxis]],1).astype(np.int32)
        segment_mask = cv2.polylines(segment_mask, [segment_poly], False, (255,0,0), 1)
        segments[theta] = segment_poly
        #img = cv2.polylines(img, [segment_poly], False, (255,0,0), 1)
        
        segment = interp_points(int(cX +np.cos(theta2)*width), int(cY-np.sin(theta2)*height), cX,cY) 
        segment_poly = np.concatenate([segment[0][:,np.newaxis], segment[1][:,np.newaxis]],1).astype(np.int32)
        #img = cv2.polylines(img, [segment_poly], False, (255,0,0), 1)
        segment_mask = cv2.polylines(segment_mask, [segment_poly], False, (255,0,0), 1)

        skull_thickness += [(theta, (segment_mask * thresh).sum())]
                
    #when theta 0, its horizontal, anticlockwise when increasing (cosx + siny), we will compute deg to rotate to vertical
    calib_deg, thickness = sorted(skull_thickness,key=lambda tup: tup[1], reverse=True)[0]
    axis = segments[calib_deg]
    img = cv2.polylines(img, [segments[calib_deg]], False, (255,0,0), 2)
    calib_deg = (90 - (calib_deg * 180 / np.pi)%180 ) # positive is anticlockwise, if axis is <90deg, we will rotate anticlockwise
    #cv2.putText(img, f"rotation {calib_deg}", (cX - 100, cY - 100),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    rot_mat = cv2.getRotationMatrix2D((cX, cY), calib_deg , 1.0)
    calib_img = cv2.warpAffine(img, rot_mat, img.shape[:-1], flags=cv2.INTER_LINEAR)
    calib_img = cv2.polylines(calib_img, [axis], False, (0,0,255), 2) #just to see original axis
    

    out = cv2.imwrite(im_path, calib_img)



def tilt_correction(im_path):
    #https://towardsdatascience.com/medical-image-pre-processing-with-python-d07694852606
    img = cv2.imread(im_path)
    im_path = str(Path(im_path.parent) / Path(im_path.stem)) + '_rot.png'
    img = img.astype(np.uint8)
    contours, hier =cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask=np.zeros(img.shape, np.uint8)

    # find the biggest contour (c) by the area
    c = max(contours, key = cv2.contourArea)

    (x,y),(MA,ma),angle = cv2.fitEllipse(c)

    cv2.ellipse(img, ((x,y), (MA,ma), angle), color=(0, 255, 0), thickness=2)

    rmajor = max(MA,ma)/2
    if angle > 90:
            angle -= 90
    else:
       angle += 96
    xtop = x + math.cos(math.radians(angle))*rmajor
    ytop = y + math.sin(math.radians(angle))*rmajor
    xbot = x + math.cos(math.radians(angle+180))*rmajor
    ybot = y + math.sin(math.radians(angle+180))*rmajor
    cv2.line(img, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (0, 255, 0), 3)

    M = cv2.getRotationMatrix2D((x, y), angle-90, 1)  #transformation matrix

    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), cv2.INTER_CUBIC)

    out = cv2.imwrite(im_path, img)




class ReportManager:
    def __init__(self, project = 'ich'):
        self.proj = project.lower()
        assert self.proj in PROJECT , f'Project {self.proj} not implemented.'
        vol_path = Path(CACHE)/ f'{self.proj}/volume.csv'
        self.vol_path = vol_path
        if not os.path.exists(Path(vol_path).parent):
            os.makedirs(Path(vol_path).parent)
        if not os.path.exists(vol_path):
            pd.DataFrame(columns=['series_id','instance_id','timestamp','target','volume']).set_index('series_id').to_csv(vol_path)


        #self.text_report_path = lambda user: Path(TARGET_DIR)/ f'report/{user}/textreport.csv'
        self.text_report_path = lambda user: Path(CACHE)/ f'report/{user}'
        #if not os.path.exists(Path(text_report_path).parent):
        #    os.makedirs(Path(text_report_path).parent)
        #if not os.path.exists(text_report_path):
        #    pd.DataFrame(columns=['user_id','series_id','timestamp','section','text']).to_csv(text_report_path)
        
        #self.text_report = pd.read_csv(text_report_path).drop_duplicates() #might be different user subset if using database
        self.load_template()

    def load_volume(self, series):
        self.store_vol = pd.read_csv(self.vol_path).drop_duplicates() #might be different user subset if using database
        report_data = {}
        series_df = self.store_vol.set_index('series_id').loc[series]
        latest = series_df['timestamp'].values.max()
        series_df = series_df.query("timestamp == @latest")
        largest_area = OrderedDict(series_df.groupby(['target','instance_id']).agg({'volume':'sum'}).reset_index().sort_values(by='volume' 
                )[['target','instance_id']].values)
        #largest_area = OrderedDict(self.ich_vol.set_index('series_id').loc[series].groupby(['target','instance_id']).agg({'volume':'sum'}).reset_index().sort_values(by='volume')[['target','instance_id']].values)

        for k,v in largest_area.items():
             if self.proj == 'ich':
                 largest_area[k] = SERVER_IP + f':{get_current_port()}'+ f'/report_im/{self.proj}/?instance={v}'
             else:
                 largest_area[k] = SERVER_IP + f':{get_current_port()}'+ f'/report_im/{self.proj}/?instance={v}&task={k}'



        report_data['Display'] = largest_area
        report_data['Volume'] = series_df.groupby('target').agg(Total=('volume',lambda x: sum(x * 0.001)), Largest=('volume',lambda x: max(x * 0.001))).to_dict()

        return report_data

    def load_text_report(self, series, user, template):
        #for detail report section 1
        assert isinstance(user, int) , 'User must be integer code.'
        rep = self.text_report_path(user)
        if not os.path.exists(Path(rep)):
            os.makedirs(Path(rep))
        if not os.path.exists(rep):
            pd.DataFrame(columns=['user_id','series_id','backend_time']).to_csv(rep, index=False)
           
           #loaded_report = pd.read_csv(rep)
           #if (series in loaded_report.query("user_id == @user")['series_id'].unique()) & (template == False):
           #    saved_query = dict(loaded_report.query("user_id == @user & series_id == @series").sort_values(by=['backend_time'], ascending=False).head(1).drop(columns='backend_time').to_dict('records')[0])
        
               #saved_query['labfindings'] = ast.literal_eval(saved_query['labfindings'])
        if (series in [p.split('.')[0] for p in os.listdir(self.text_report_path(user))]) & (template == False):
            with open(self.text_report_path(user) / f'{series}.json', 'r') as f:
                saved_query = json.load(f)
        else:
            saved_query = {}
            saved_query['content'] = self.template

        return saved_query
            
    def update_text_report(self,series,user,obj):
        ct = datetime.datetime.now()
        #content = obj['content']
        #assert 'series' in content, 'Series ID not specified in POST Content.'
    
        # Current save as json, so will just replace with same name
        #content['backend_time'] = ct
        #content['user_id'] = user
        #content['series_id'] = series
        
        with open(self.text_report_path(user) / f'{series}.json', 'w') as f:
             json.dump(obj, f)
       
           #existing = pd.read_csv(self.text_report_path(user))
           #updated = pd.concat([existing, pd.DataFrame([content])])
           #updated.to_csv(self.text_report_path(user), index=False)
           #pd.DataFrame([content]).to_csv(self.text_report_path(user_id), mode='a', index=False)

        
        

    def reload(self):
        self.__init__()

    def load_template(self):
        # template allow for user upload, save into user database
        with open('template/text_report.yaml', 'r') as stream:
             try:
                 self.template = yaml.safe_load(stream)

             except yaml.YAMLError as exc:
                 raise exc






class InputManager:
    def __init__(self, 
                 args, 
                 fetcher,
                 project : str = 'ICH',
                 modality : str = 'CT',
                 lookup = True,
                 save_preprocess = True,
                 parallel = False):
        
        self.args = args
        self.fetcher = fetcher
        self.which_pipeline(modality)
        self.project = project.lower()
        assert self.project in PROJECT , f'Project {project} Not Implemented.'
        self.lookup = lookup
        self.parallel = parallel
        self.save_preprocess = save_preprocess
        self.process()

    def which_pipeline(self, modality : str):
        existing = {
                     'ct' : CT_Pipeline
                      
                   }
        assert isinstance(modality , str) , 'arg `Modality` must be string'
        if not modality.lower() in existing:
            raise NotImplementedError(f'Current pipeline only accept one of {list(existing.keys())}')
        self.pipeline = existing[modality.lower()]


   


    def process(self):	
        
        '''
        The pipeline will perform the usual preprocessing of modality, 
        however we also allow for user input so we will need another layer to bridge user input and pipeline.

        '''

        
        dcmfile, pseudo_name = self.fetcher(self.args)


        ##############################
        ## Lookup for cached
        if self.lookup:
            frontend_json = {}
            for id_ in pseudo_name['instanceID']:

                path = self.database_lookup( id_)

                if path is not None:
                    frontend_json[id_] = path
                else: # if not cached
                    frontend_json = {}
                    break
            if len(frontend_json) == len(pseudo_name['instanceID']):
                self.frontend_json = frontend_json
                self.dcmfile = None
                return

        ##############################



        if isinstance(self.pipeline, CT_Pipeline):
            if ('wl' in args) & ('ww' in args):
                ww = args.get('ww')
                wl = args.get('wl')
                ww = int(ww) if ww is not None else ww
                wl = int(wl) if wl is not None else wl
            else:
                ww = 80
                wl = 40
            
            with self.pipeline(ww=ww,wl=wl, parallel=self.parallel) as pipe:
                self.dcmfile = pipe(dcmfile)

        else:
            with self.pipeline(parallel=self.parallel) as pipe:
                self.dcmfile = pipe(dcmfile)
            

        ###DicomTensor should carry orthanc ID 
        #from matplotlib import pyplot as plt
        for obj, val in zip(self.dcmfile, pseudo_name['instanceID']):
            obj.attr('orthancID', val)
            obj.attr('orthancSeriesID', pseudo_name['seriesID'])

            #plt.imshow(obj.im)
            #plt.savefig(f'temp/{val}.png')

            if self.save_preprocess:
                cache_path = str(Path(CACHE) / self.project / val)

                if not os.path.exists(Path(cache_path)): #check dir
                    os.makedirs(Path(cache_path))

                ##====================================
                #get preprocessed image here
                x_path = str(Path( cache_path))+  '_inter.png'
                img = Image.fromarray(obj.vis.astype(np.uint8))
                img.save(x_path, cmap='gray')
                os.system(f"chgrp anat {x_path}")
                ##====================================


        
 



    def database_lookup(self, inst_id):
    
        assert isinstance(self.project, str) , '`Project` should be string'
        assert self.project.lower() in PROJECT , f'Project {self.proj} not implemented.'
 
        #TODO in future, ich will output mask by subtype like tissueseg, not sum together, currently we split 2 conditions


        if self.project == 'ich':
            path = str(Path(CACHE) / f'{self.project.lower()}/{inst_id}.npy')
            url = f'{SERVER_IP}:{get_current_port()}/retrieve_np/{self.project.lower()}/?instance={inst_id}'

    
            if os.path.exists(path):
                return url
    
            else:
                return None

        else:
            cls_url = {}
            path = str(Path(CACHE) / f'{self.project.lower()}/{inst_id}_*.npy')
            for p in glob.glob(path):
                cls = p.split(inst_id+'_')[1][:-4]
                url = f'{SERVER_IP}:{get_current_port()}/retrieve_np/{self.project.lower()}/?instance={inst_id}&task={cls}'
                if os.path.exists(p):
                    cls_url[cls] = url
                else:
                    return None

            return cls_url

    



class OutputManager:

    def __init__(self):
        pass

    def postprocess(self,*args):
        self.vol = Volume(self.project)
        project_fn = {
                  'ich' : self.process_ich,
                  'tissueseg' : self.process_tissueseg

                }
        return project_fn[self.project](*args)



    def process_ich(self, pred : Dict , mask : Dict):

        '''
        Convert model output to json, in YOLO+EM number mask and detection bbox may not be equal 

        Args:
          pred: {uid1: [ [x1,y1,x2,y2, class] , [x1,y1,x2,y2 , class] ] }
          mask: {uid1: [ numpy.array [512 x 512] , ... ]}


        Output:
          { InstanceID:
                          { Subtype1: { url: str , bbox: List} },... }
        '''
        label_names= ['chronic','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']
        main_json = {}
        complete_mask = {}
        is_series = len(self.dcmfile) > 1
        curr_time = datetime.datetime.now()
        masks3D = []
        for uid in pred.keys(): #uid is the ID in dicom image
            dcmobj = [t for t in self.dcmfile if t.SOPInstanceUID == uid][0]
            seriesID = dcmobj.orthancSeriesID
            save_uid = dcmobj.orthancID #save_uid is the ID used in dicom server - frontend - backend
            main_json[save_uid] = {}
            
            cache_path = str(Path(CACHE) / self.project / save_uid)

            #if not os.path.exists(Path(cache_path)): #check dir
            #    os.makedirs(Path(cache_path))

            ###====================================
            ##get preprocessed image here
            #x_path = str(Path( cache_path))+  '_inter.png'
            #img = Image.fromarray(dcmobj.vis.astype(np.uint8))
            #img.save(x_path, cmap='gray')
            ##====================================

            ## declare a master mask
            shape = dcmobj.vis.shape 
            M = np.zeros(shape).astype(np.int)
            M_bb = np.zeros(shape).astype(np.uint8)
         


            #========  PROCESS SEGMENTATION MASK AND BBOX ========#
            #### EM has an issue doesnt return mask for such bbox or slice
            if uid not in mask:
                pass
            else:
                for i, label in enumerate(label_names): #look for subtype
                    subtype_bbox_mask = [(bbox,arr ) for j,(arr, bbox) in enumerate(mask[uid]) if bbox[-1]==i] #slice bbox of the subtype, and corresponding mask
                    subtype_dict = {}

                    #update first time so subtype that doesnt exist will also appear
                    if is_series: 
                        self.vol.update((dcmobj, label, 0), curr_time)

                    for k, (bb, m) in enumerate(subtype_bbox_mask): #Subtype1-(k)
                    ## adding the mask

                        M += m

                        # bbox mask
                        x1,y1,x2,y2 = bb[:4].astype(int)
                        M_bb = cv2.rectangle(np.array(M_bb), (x1, y1), (x2, y2), (255,0,0), 2)
                        #cv2.putText(M_bb, label, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 0.7, (36,255,12), 2)
                        cv2.putText(M_bb, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

                        subtype_dict[k] = {}
                        # save im at url , path
                        subtype_dict[k]['url'] = os.path.join(cache_path, f'{label}_{k}.png')
                        #matplotlib.image.imsave(subtype_dict[k]['url'], m)
                        #os.system(f"chgrp anat {subtype_dict[k]['url']}")
                        #bbox
                        subtype_dict[k]['bbox'] = [str(int(x)) for x in bb[:4]]
                        #confidence
                        subtype_dict[k]['conf'] = str(bb[-2])
                        
                        if is_series:
                            self.vol.update((dcmobj, label, (m>0).sum()), curr_time)



                    main_json[save_uid][label] = subtype_dict

            assert M.shape == dcmobj.vis.shape , 'Mask Shape must align with input dicom shape'

            #============ Save Instance Mask =================
            #np.save(cache_path+'.npy',(M>0).astype(np.uint8) )
            #os.system(f"chgrp anat {cache_path + '.npy'}")
            #=================================================

                #convert_png_transparent(cache_path + '.png')
            complete_mask[save_uid] = f'{SERVER_IP}:{get_current_port()}/retrieve/ich/?instance={save_uid}'

                #save bbox mask
            cv2.imwrite(cache_path + '_bb.png',M_bb )
            convert_png_transparent(cache_path + '_bb.png')
            os.system(f"chgrp anat {cache_path + '_bb.png'}")



            frontend_req = []
            for b in pred[uid]:
                j  = {
                        'start': 
                            {'x': str(b[0]),
                             'y': str(b[1])
                            }, 
                        'end' :
                            {'x': str(b[2]),
                             'y': str(b[3])
                            }
                    }
                dic = {}
                dic['text'] = label_names[int(b[-1])]
                dic['handles'] = j
                frontend_req += [dic]


            with open(cache_path+ '.json', 'w') as f:
                json.dump( frontend_req, f, indent=4, sort_keys=False)
            os.system(f"chgrp anat {cache_path + '.json'}")


            masks3D += [M[np.newaxis,...]]
            ##with open(os.path.join(os.path.dirname(cache_path), 'meta.json'), 'w') as f:
            #with open(cache_path+ '.json', 'w') as f:
            #    json.dump(main_json[save_uid], f, indent=4, sort_keys=False)
            #os.system(f"chgrp anat {cache_path + '.json'}")


        if is_series: 
            self.vol.flush()

        #============= Save Series 3D masks ==================
        cache_path = str(Path(CACHE) / self.project / seriesID)
        masks3D = np.concatenate(masks3D)
        np.save(cache_path+'.npy',(masks3D>0).astype(np.uint8) )
        os.system(f"chgrp anat {cache_path + '.npy'}")
        #=====================================================
            
        return main_json, complete_mask


    def process_tissueseg(self, mask : Dict):
        '''
        Current output 

        { 
          'wm' : {
                    'Instance1' : Array,
                    ...

                 },
          'csf' : {...}

        } 

        '''
        output = defaultdict(dict)
        is_series = len(self.dcmfile) > 1
        curr_time = datetime.datetime.now()

        for cls in mask.keys():
            for subj, arr in mask[cls].items():
                dcmobj = [t for t in self.dcmfile if t.orthancID == subj][0]
                cache_path = Path(CACHE) / self.project / f'{subj}_{cls}.npy'
                np.save( cache_path , (arr > 0).astype(np.uint8))
                os.system(f"chgrp anat {cache_path}")
                cache_path = Path(CACHE) / self.project / f'{subj}_{cls}.png'
                matplotlib.image.imsave(cache_path, arr)
                output[subj][cls] = f'{SERVER_IP}:{get_current_port()}/retrieve/tissue/?instance={subj}&task={cls}'

                if is_series:
                    self.vol.update((dcmobj, cls, (arr>0).sum()), curr_time)

        if is_series: 
            self.vol.flush()

            self.generate_plot()

        return output


    def generate_plot(self):
        
        series = self.args.get('series_id')
        #data = reporter_tissueseg.load_volume(series)
        vol = pd.DataFrame(self.vol.df).groupby('target').agg(volume=('volume',lambda x: sum(x * 0.001))).to_dict()['volume']
        classes = {
                  'ticv' : ['bet'],
                  'tbv' : ['gm','wm'],
                  'vent' : ['vent'],
                }
    
        fig, ax = plt.subplots(1, len(classes), figsize=(20,5))
    
    
        hcp_path = '/home/data_repo/cache/vol_hcp.csv'
        kcl_path = '/home/data_repo/cache/vol_kcl.csv'
        aomic_path = '/home/data_repo/cache/vol_aomic.csv'
        
        hcp = pd.read_csv(hcp_path)
        #kcl = pd.read_csv(kcl_path)
        #aomic = pd.read_csv(aomic_path)
        
        hcp.rename(columns={'Subject': 'sub_id', 'Age': 'age', 'Gender': 'gender'}, inplace=True)
        hcp['sub_id'] = 'HCP_' + hcp['sub_id'].astype('str')
        
        hcp_grouped = hcp.groupby(['age']).describe()
    
    
        for i, c in enumerate(classes):
            query = sum(list(map( lambda x: vol[x], classes[c] )))
            age_raw = self.dcmfile[0].PatientAge
            #age_raw = self.dcmfile[0][('0010','1010')] # series so just get from first dcm, 022Y
            age = int(age_raw[:-1])
            age_range = age_band(age)
            sns.lineplot(data=hcp_grouped[c][['25%', '50%', '75%']], ax=ax[i])
            ax[i].plot(age_range, query, 'rp') # https://stackoverflow.com/a/46637880
            ax[i].set_title(c.upper())
    
    
    # need to get vol and age range
    
        plt.savefig(Path(CACHE)/ 'tissueseg' / f'distplot_{series}.png')
        plt.close()





def age_band(age):
    if age >= 22 and age <= 25:
        return '22-25'
    elif age >= 26 and age <= 30:
        return '26-30'
    elif age >= 31 and age <= 35:
        return '31-35'
    else: # will put <22 here too... 
        return '36+'




class Manager(InputManager, OutputManager):


    def __init__(self, args, fetcher, **kwargs):
        super().__init__(args,fetcher ,**kwargs)
        


    def __enter__(self):
        return self


    def __exit__(self,exception_type, exception_value, traceback):
        pass




class MaskAnnotater:
    ## report_im endpoint will query for instance image, if updated by new annotation, needs to update in numpy
    def __init__(self, project='ich'):
        self.key = 'segMasksArrayBuffer'
        self.project = project

    def proc_im(self, json):
        target = json[self.key]
        shape = target['shape']
        arr = np.resize(np.array(list(target['data'].values())), shape)
        return arr

    def proc_series(self,list_objects):
        new_mask = {}
        for obj in list_objects:
            assert 'instanceNumber' in obj , 'Object doesnt have instance ID'
            new_mask[obj['instanceNumber']] = self.proc_im(obj)
        return new_mask

    def save_im(self,inst_id ,arr ):
        arr = arr.astype(np.uint8)
        im_path = Path(CACHE) / self.project / inst_id
        assert os.path.exists(str(im_path)+ '.npy'), 'Instance Mask must exists before.'
        save_path = str(im_path) + 'edit' + '.npy'
        np.save(save_path, arr)
        os.system(f"chgrp anat {save_path}")

    def __call__(self, user_id, series_id, objects):
        new_mask = self.proc_series(objects)
        for k ,v in new_mask.items():
            self.save_im(k, v)



class BoxAnnotater:
    def __init__(self, project='ich', **kwargs):
        self.project = project
        self.opt = kwargs

    def __call__(self, user_id : str, series_id : str, objects: List[Dict] ):
        new_box = self.proc_content(objects)
        
        if self.opt['inference'] == True:
            #assert len(objects) > 1 , 'Inferencing bbox is only allowed once at a time'
            self.inference(objects)

    def proc_content(self, list_objects):
        if not isinstance(list_objects, list):
            list_objects = [list_objects]
        for obj in list_objects:
            assert 'instanceNumber' in obj , 'Json doesnt contain instanceNUmber'
            boxes = obj['bbox']


            cache_path = str(Path(CACHE) / self.project / obj['instanceNumber'])
            with open(cache_path+ 'edited.json', 'w') as f:
                json.dump( boxes,  f, indent=4, sort_keys=False)
            os.system(f"chgrp anat {cache_path + '.json'}")

    def save(self,inst_id, obj): #save as json
        pass

    def inference(self, objects):
        inference_mask = []
        for obj in self.objects:
            inst_id = obj['instanceNumber']
            preprocessed_slice = Image.open(Path(cache)/ str(inst_id + '_inter.png'))
            #clsutered = EM_segm( preprocessed_slice, obj['BoundingBox'])
            inference_mask += [clustered] #or save them in tmp , but how do we know if this should be updated to real mask




def Annotater( project : str ,task: str):
    '''
    project: The project which the request is from,  'ich' or 'tissueseg'
    task   : To specify what annotation task the request is from, 'mask' or 'box', 
    objects: A List of json, can be just for one instance/slice
    '''

    if task.lower() == 'mask':
        return MaskAnnotater(project)

    elif task.lower() == 'box':
        return BoxAnnotater(project)

    else:
        raise NotImplementedError 

