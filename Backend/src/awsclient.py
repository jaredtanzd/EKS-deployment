

import boto3
from io import BytesIO
import yaml
from botocore.errorfactory import ClientError
import pandas as pd
import PIL
from PIL import Image
import json
import cv2
import numpy as np
from pathlib import Path

class Bucket():
    def __init__(self, mount_path=''):
        '''
        mount_path : Path to create or mount on s3 bucket , for search , list action. Save object does not require a mount_path

        '''
        self.mount = Path(mount_path)
        cfg_path = Path(__file__).parent / 'credentials.yaml'
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        self.bucket_name = cfg['aws_bucketalias']
        session = boto3.Session()
        self.s3_client = session.client("s3")

        self.f = BytesIO()
        self.g_fn = lambda filename : self.s3_client.download_fileobj(self.bucket_name, filename, self.f)
        self.dl_file = lambda source,dest : self.s3_client.download_file(self.bucket_name, source, dest)

    def exists(self, path, get=False):
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=path)
            if get:
                # in this case we are writing from s3 to current fs , so check os path
                os.makedirs(str(self.mount/ path) , exist_ok=True)
                self.dl_file(path , str(self.mount / path)) # download to dest, same path as s3 at ~

        except ClientError as e:
            return None
        return response

    def get(self, path, lib=None):
        # into memory as object, does not write into disk
        self.exists(path)
        if lib is None:
            if '.png' in path:
                lib = 'pillow'
            elif '.npy' in path:
                lib = 'numpy'
            else:
                pass

        if '.json' in path:
            filecontent = self.s3_client.get_object(Bucket=self.bucket_name, Key=path)['Body'].read().decode('utf-8')
            return json.loads(filecontent)

        self.g_fn(path)
        obj = debinarize(self.f, lib)
        return obj


    def save(self, obj , path): # only binary, if for file, use upload_file

        try:
            if '.json' in path:
                self.s3_client.put_object(Body=json.dumps(obj), Bucket=self.bucket_name, Key=path)
                ### tested 
                
                #json_obj = bucket.get('meta.json')
                #bucket.save(json_obj,'meta2.json')
                #json_obj2 = bucket.get('meta2.json')
                #json_obj == json_obj2

                ###
                return
            #object_ = self.s3_client.Object(self.bucket_name, path)
            obj = binarize(obj)
            self.s3_client.put_object(Body=obj, Bucket=self.bucket_name, Key=path)
            # obj must be in binary, must have read fn

            #TODO check if Image from array save maintain single channel

        except Exception as e:
            print(e)


    def upload(self, source_path, dest_path ):
        pass

 

    def list(self,prefix, get=False):
        ''' 
        Prefix should be the beginning pattern of the name 
        '''
        response = self.s3_client.list_objects(Prefix=prefix, Bucket=self.bucket_name)
        if 'Contents' not in response: return None
        objs = []
        for obj in response['Contents']:
            if get:
                path = obj['Key']
                os.makedirs(str(self.mount/ path) , exist_ok=True)
                self.dl_file(path , str(self.mount / path)) # download to dest, same path as s3 at ~
            objs += [obj['Key']]
        return objs


def debinarize(byte_obj, lib):
    types = ['numpy','pillow','cv2']
    assert lib in types, f'Specify library in {types}'
    assert isinstance(byte_obj, BytesIO), 'Object must be BytesIO'
    if lib == 'cv2':
        return cv2.imdecode(np.asarray(bytearray(byte_obj.getvalue())), cv2.IMREAD_COLOR) #for rgb image, used in rotation_lib
    elif lib == 'pillow':
        return Image.open(byte_obj)
    elif lib == 'numpy':
        byte_obj.seek(0)
        return np.load(byte_obj,allow_pickle=True)
    else:
        raise NotImplementedError


def binarize(obj):
    f = BytesIO()
    if isinstance(obj, PIL.Image.Image):
        obj.save(f, format= 'PNG')
        f.seek(0)
    elif isinstance(obj,np.ndarray ):
        np.save(f, obj, allow_pickle=True)
        f.seek(0)
    else:
        pass
    return f



class Dummy():
    def __init__(self, *args, **kwargs):
        pass

    def exists(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass

    def list(self, *args, **kwargs):
        pass

    def get(self, *args, **kwargs):
        pass

    def upload(self, *args, **kwargs):
        pass




if __name__ == '__main__':
    bucket = Bucket('/tmp/')
    obj = np.array([[1,1,1],[1,1,1]])
    bucket.save(obj, '/tmp/testarray.npy')
    obj_ret = bucket.get('/tmp/testarray.npy')


