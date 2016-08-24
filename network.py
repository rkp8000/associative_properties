from __future__ import division, print_function
import numpy as np


class Basic(object):
    """
    Basic discrete time network with random connections between item and memory units.
    """
    
    def _make_weight_matrix(self):
        
        w = np.random.rand(self.n_item_units, self.n_mem_units) < self.p_cxn
        
        return w.astype(float)
    
    def __init__(self, n_item_units, n_mem_units, p_cxn):
        
        self.n_item_units = n_item_units
        self.n_mem_units = n_mem_units
        self.p_cxn = p_cxn
        
        # make weight matrix
        
        self.w = self._make_weight_matrix()
        self.w_t = self.w.T
        
    def run(self, drives, record=None):
        """
        Present a sequence of stimuli/drives to the network.
        
        :param drives: sequence of drives (each is a dict)
        :param record: list of variables to record -- options are:
            'input_item' -- inputs to item units
            'input_mem' -- inputs to memory units
            'atvn_item' -- item unit activations
            'atvn_mem' -- memory unit activations
            'hx_mem' -- memory unit hyperexcitabilities
        """
        
        if record is None:
            
            record = []
            
        record = {variable:[] for variable in record}
            
        # set initial states
        
        atvn_item = np.zeros((self.n_item_units, 1))
        atvn_mem = np.zeros((self.n_mem_units, 1))
        hx_mem = np.zeros((self.n_mem_units,))
        
        # loop through drive sequence
        
        for drive in drives:
            
            drive_item = np.zeros((self.n_item_units,))
            
            if drive['type'] == 'item_drive':
                
                for item_idx in drive['params']:
                    
                    drive_item[item_idx] = np.inf
            
            elif drive['type'] == 'item_inhibition':
                
                drive_item[:] = -np.inf
            
            # get inputs to items and memory
            
            input_item = self.w.dot(atvn_mem).flatten() + drive_item
            input_mem = self.w_t.dot(atvn_item).flatten() + hx_mem
            
            # store inputs if desired
            
            if 'input_item' in record.keys():
                
                record['input_item'].append(input_item)
                
            if 'input_mem' in record.keys():
                
                record['input_mem'].append(input_mem)
                
            # get 2-winner-take-all item activations
            
            n_positive_inputs_items = (input_item > 0).sum()
            
            atvn_item = np.zeros((self.n_item_units, 1))
            
            if n_positive_inputs_items >= 1:
                
                input_item_argmax = input_item.argmax()
                atvn_item[input_item_argmax, 0] = 1
                input_item[input_item_argmax] = 0
                
            if n_positive_inputs_items >= 2:
                
                input_item_argmax = input_item.argmax()
                atvn_item[input_item_argmax, 0] = 1
                
            # get memory activations
            
            atvn_mem = np.zeros((self.n_mem_units, 1))
            atvn_mem[input_mem >= 2, 0] = 1
            
            # get memory hyperexcitabilities
            
            hx_mem[atvn_mem[:, 0].astype(bool)] = 1
            
            # store activations if desired
            
            if 'atvn_item_idxs' in record.keys():
                
                record['atvn_item_idxs'].append(tuple(atvn_item.flatten().nonzero()[0]))
                
            if 'atvn_item' in record.keys():
                
                record['atvn_item'].append(atvn_item)
                
            if 'atvn_mem' in record.keys():
                
                record['atvn_mem'].append(atvn_mem)
                
            if 'hx_mem' in record.keys():
                
                record['hx_mem'].append(hx_mem)
                
        return record