# -*- coding: utf-8 -*-




import numpy as np
import pandas as pd

import pickle
import os
from datetime import datetime
from .target import Target 




# pickle output and save it in dictionary structure
def store_output(obj, collection=None, destination=None, notes=None, str_to_txt=False):
    
    if destination is None:
        
        destination = os.getcwd()
        
    else:
        destination = os.path.abspath(destination)
        
    
    target = obj.target
        
    if isinstance(target, Target):
        target_name = target.name.replace(' ', '')
        target_logname = target.name.replace(' ', '')
        
    else:
        target_name = target.replace(' ', '')
        target_logname = target.replace(' ', '')
        
        
    date = datetime.today().strftime('%Y%m%d')
        
    
    if collection is not None:
        
        collection = collection.replace(' ', '')
        
        dirpath = os.path.join(destination, collection, target_name, date)
        
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        
        i = 0
        while True:
            
            file = f'{target_name}.{i}'
                
            
            filepath = os.path.join(dirpath, file)
            
            
            if os.path.exists(filepath):
                i += 1
                continue
            
            break
        
        
        with open(filepath, 'wb') as f:
            
            pickle.dump(obj, f)
        

        logfile = os.path.join(destination, collection, r'_log.txt')
        
        if os.path.exists(logfile):
            
            log_array = np.loadtxt(logfile, dtype=str)
            
            try:
                log = pd.DataFrame(log_array[:, 1], index=pd.Index(log_array[:, 0], name='target'), columns=['date'])
            
            except IndexError:
                # if there is only 1 entry in the log table the indexing will be different
                log = pd.DataFrame([log_array[1]], index=pd.Index([log_array[0]], name='target'), columns=['date'])
            
        else:
            
            log = pd.DataFrame([date], index=pd.Index([target_logname], name='target'), columns=['date'])
            
            
        if target_logname in log.index:
            log.update(pd.DataFrame([date], index=pd.Index([target_logname], name='target'), columns=['date']))
        
        else:
            log = log.append(pd.DataFrame([date], index=pd.Index([target_logname], name='target'), columns=['date']))
        
        
        np.savetxt(logfile, log.reset_index().values, header=f"Most recent run date (YYYYMMDD) for each target. Updated {datetime.today().strftime('%Y-%m-%d')} (YYYY-MM-DD). \n", fmt='%s', delimiter='\t')
        
        
    else:
        
        dirpath = os.path.join(destination, target_name, date)
        
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        
        i = 0
        while True:
            
            file = f'{target_name}.{i}'
                
            
            filepath = os.path.join(dirpath, file)
            
            
            if os.path.exists(filepath):
                i += 1
                continue
            
            break
        
        
        with open(filepath, 'wb') as f:
            
            pickle.dump(obj, f)
            
    
    if notes is not None:
        
        with open(filepath+'.notes.txt', 'w') as f:
            f.writelines(notes)
            
    
    if str_to_txt:
        
        with open(filepath+'.txt', 'w') as f:
            f.writelines(obj.__str__())
            
    
    return filepath




# retrieve output data pickled via store_output
def extract_output(name, location=None, collection=False):
    
    name = name.replace(' ', '')
    
    if location is None:
        
        location = os.getcwd()
        
    else:
        location = os.path.abspath(location)
        
    
    if collection:
        
        collection_path = os.path.join(location, name)
        
        
        collection_contents = os.listdir(collection_path)
        
        targets = collection_contents
        
        try:
            targets.remove('.DS_Store')
        except:
            pass
        
        try:
            targets.remove('_log.txt')
        except:
            pass
        
        logfile = os.path.join(collection_path, '_log.txt')
        
        log = np.loadtxt(logfile, dtype=str, delimiter='\t')
        
        try:
            log_df = pd.DataFrame(log[:, 1], columns=['date'], index=pd.Index(log[:, 0], name='target'))
        
        except IndexError:
            log_df = pd.DataFrame([log[1]], columns=['date'], index=pd.Index([log[0]], name='target'))
        
        
        data = {target:None for target in targets}
        
        for target in targets:
            
            date = log_df.loc[target, 'date']
            
            filedir = os.path.join(collection_path, target, date)
            
            filedir_contents = os.listdir(filedir)
            
            try:
                filedir_contents.remove('.DS_Store')
            except:
                pass
            
            filedir_contents.sort()
            
            n = 0
            for file in filedir_contents:
                    
                if f'.{n}' in file or f'.R{n}' in file:
                    
                    continue
                
                else:
                    
                    n += 1
                    continue
                
            
            pickle_file = os.path.join(filedir, f'{target}.{n}')
            
            with open(pickle_file, 'rb') as f:
            
                obj = pickle.load(f)
                
            
            data.update(
                {
                    target : obj
                    }
                )
            
            
        return data
    
    
    else:
        
        target_dir = os.path.join(location, name)
        
        dates = os.listdir(target_dir)
        
        try:
            dates.remove('.DS_Store')
        except:
            pass
        
        date = max(dates)
        
        
        filedir = os.path.join(target_dir, date)
        
        filedir_contents = os.listdir(filedir)
        
        try:
            filedir_contents.remove('.DS_Store')
        except:
            pass
            
        filedir_contents.sort()
        
        n = 0
        for file in filedir_contents:
                
            if f'.{n}' in file or f'.R{n}' in file:

                continue
            
            else:
                
                n += 1
                continue
        
        pickle_file = os.path.join(filedir, f'{name}.{n}')
        
        with open(pickle_file, 'rb') as f:
        
            obj = pickle.load(f)
            
        
        data = obj
        
        
        return data
        
        
    
    
    
    




