import os
import requests

from functools import partial

import json
import yaml
import socket
import io
import pydicom
from pathlib import Path
import getpass



def gethost():
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    return hostname , ip




def get_credentials(cfg_path):
 
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    
    user = input("Username:")
    pwd = getpass.getpass('Password:')
    orthanc_ip = 'https://' + gethost()[1] +'/orthanc' #same ip to backend server
    ip_addr = 'ip_addr_aws' if 'bionet' not in gethost()[0] else 'ip_addr'
    cert_path = 'cert_path_aws' if 'bionet' not in gethost()[0] else 'cert_path'
    response = requests.get(cfg[ip_addr], verify=cfg[cert_path], auth=(user,pwd))
    if response.status_code != 200: return response

    #return partial(lambda x: requests.get(os.path.join(cfg['ip_addr']+'/orthanc',x ), verify=cfg['cert_path'], auth=(user,pwd)))
    return partial(lambda x: requests.get(os.path.join(cfg[ip_addr],x ), verify=cfg[cert_path], auth=(user,pwd)))


class Fetcher:

    def __init__(self, cfg_path):
        #look into wd to get username password and certificate
        self.fetch_fn = get_credentials(cfg_path)

    def _get_item(self, kwargs):
        series = kwargs.get('series',None)
        instance = kwargs.get('instance',None)
        fseries = lambda x: '' if x is None else 'series/'+str(x)
        finst = lambda x: '' if x is None else 'instances/'+str(x)
        query_path = lambda instance : os.path.join(finst(instance), 'file')
        pseudo_name = {}

        if all([arg is None for arg in (series, instance)]):
            #simply get first isntance
            result = self.fetch_fn('instances/')
            inst_id = result._content.decode('latin-1').split('\"')[1] #TODO [1] is the first inst_id, but need a better way to split the byte
            result = self.fetch_fn(query_path(inst_id))
            pseudo_name['instanceID'] = [inst_id]
            pseudo_name['seriesID'] = series


        elif (series is not None) & (instance is None): #fetch all instances
            instance_list = os.path.join(fseries(series) , 'instances')
            result_list = self.fetch_fn(instance_list)
            result_list = json.loads(result_list.text)
            list_id = [l['ID'] for l in result_list]
            orthanc_series_index = [l['IndexInSeries'] for l in result_list]
            result_list = [tup[1] for tup in sorted(zip(orthanc_series_index,list_id), key=lambda tup: tup[0])]
            pseudo_name['instanceID'] = result_list
            result = list(map(lambda x: self.fetch_fn(query_path(x)) , result_list))
            pseudo_name['seriesID'] = series
        
        else: #instance not None, get instance id
            result = self.fetch_fn(query_path(instance))
            pseudo_name['instanceID'] = [instance]
            pseudo_name['seriesID'] = None

        return result, pseudo_name

    def _get_dicom(self, result):
        if isinstance(result , list):
            dcmfile = []
            for obj in result:
                dcm_byte = io.BytesIO(obj.content)
                try:
                    dcm = pydicom.dcmread(dcm_byte)
                except pydicom.errors.InvalidDicomError:
                    dcm = pydicom.dcmread(dcm_byte, force=True)

                dcmfile += [dcm]
            
            #dcmfile = sorted(dcmfile, key=lambda s: s.SliceLocation) # make sure that dcm files are in correct order

        else: #single dcm 
            dcm_byte = io.BytesIO(result.content)
            try:
                dcmfile = pydicom.dcmread(dcm_byte)
            except pydicom.errors.InvalidDicomError:
                dcmfile = pydicom.dcmread(dcm_byte, force=True)

            dcmfile = [dcmfile]

        return dcmfile
           
     
    def __call__(self, kwargs):
     
        result, pseudo_name = self._get_item(kwargs) #check iterable
        dcmfile = self._get_dicom(result)
        return dcmfile, pseudo_name



#fetcher = Fetcher(Path(__file__).parent.parent / 'src/credentials.yaml')
#assert callable(fetcher.fetch_fn), f'Wrong ID and Password. {fetcher.fetch_fn}'
