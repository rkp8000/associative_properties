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


class LIFWithKWTAItems(object):
    """
    Continuous time network of LIF neurons with an external mechanism that allows at most
    2 item neurons to be active at a time. Synapses are conductance-based.

    :param tau: membrane time constant
    :param v_rest: resting potential
    :param v_th: spike-threshold potential
    :param v_reset: reset potential
    :param t_rp: refractory period
    :param n_items: number of item neurons
    :param n_assocs: number of association neurons
    :param tau_fre: time constant for estimating firing rate
    :param k_wta: number of allowed active neurons in k-WTA rule
    :param wta_th: winner-take-all threshold
    :param wta_inh: inhibition level to implement WTA (should be negative)
    :param noise: noise level
    :param w_pa: connectivity matrix from association neurons to principal neurons
    :param w_ap: from principal to association neurons
    :param w_am: from memory to association neurons
    :param w_ma: from association to memory neurons
    :param w_mm: from memory to memory neurons
    """

    def __init__(
            self, tau, v_rest, v_th, v_reset, v_min, t_rp,
            v_rev_syn_exc, tau_syn_exc,
            tau_fre, k_wta, wta_th, wta_inh, noise,
            w_pa, w_ap, w_am, w_ma, w_mm):

        self.tau = tau
        self.v_rest = v_rest
        self.v_th = v_th
        self.v_reset = v_reset
        self.v_min = v_min
        self.t_rp = t_rp

        self.v_rev_syn_exc = v_rev_syn_exc
        self.tau_syn_exc = tau_syn_exc

        n_items = w_pa.shape[0]
        n_assocs = w_pa.shape[1]
        n_mems = w_mm.shape[0]

        self.n_items = n_items
        self.n_assocs = n_assocs
        self.n_mems = n_mems

        self.item_idxs = np.arange(n_items)
        self.assoc_idxs = np.arange(n_items, n_items + n_assocs)
        self.mem_idxs = np.arange(n_items + n_assocs, n_items + n_assocs + n_mems)

        self.n_neurons = n_items + 2 * n_assocs

        self.tau_fre = tau_fre
        self.k_wta = k_wta
        self.wta_th = wta_th
        self.wta_inh = wta_inh
        self.noise = noise

        self.w_pa = w_pa
        self.w_ap = w_ap
        self.w_am = w_am
        self.w_ma = w_ma
        self.w_mm = w_mm

    def flatten_drive(self, drive):
        """
        Convert a dict drive into a 1D array.

        :param drive: dictionary of drives to items, assocs, and mems
        :return: 1D drive array to all neurons
        """

        drive_flat = np.zeros((self.n_neurons,))

        for key, val in drive.items():

            if key == 'item':

                drive_flat[self.item_idxs] = val

            elif key == 'assoc':

                drive_flat[self.assoc_idxs] = val

            elif key == 'mem':

                drive_flat[self.mem_idxs] = val

            else:

                raise Exception('Unrecognized neuron type "{}".'.format(key))

        return drive_flat

    def calc_synaptic_conductances(self, g, spikes, dt):
        """
        Calculate the synaptic conductances given the previous conductances and the current spikes.

        :param g: previous conductances for all neurons
        :param spikes: previous spikes
        :return: new conductances
        """

        items = self.item_idxs
        assocs = self.assoc_idxs
        mems = self.mem_idxs

        syn_inputs = np.nan * np.zeros((self.n_neurons,))

        syn_inputs[items] = self.w_pa.dot(spikes[assocs])
        syn_inputs[assocs] = self.w_ap.dot(spikes[items]) + self.w_am.dot(spikes[mems])
        syn_inputs[mems] = self.w_ma.dot(spikes[assocs]) + self.w_mm.dot(spikes[mems])

        dg = (dt / self.tau_syn_exc) * (-g + (syn_inputs * (self.tau_syn_exc / dt)))

        return g + dg

    def calc_voltage(self, v, g, drive, noise, dt):
        """
        Calculate the new voltage given the previous voltage and all inputs.

        :param v: previous voltage
        :param g: synaptic conductances for all neurons
        :param drive: drive to all neurons
        :param spikes: spikes from all neurons
        :param noise: noise to all neurons
        :return: voltage for all neurons
        """

        # total inputs are drive currents + synaptic currents

        inputs = self.flatten_drive(drive) + (g * (self.v_rev_syn_exc - v))

        dv = (dt / self.tau) * (-(v - self.v_rest) + inputs + noise)

        return v + dv

    def run(self, drives, v_init, g_init, dt, record=None):
        """
        Run the network given a set of drives.

        :param drives: dictionary of drives to apply at each time point with keys
            "item", "assoc", or "memory"
        :param v_init: initial voltages for all neurons
        :param g_init: initial conductances of all neurons
        :param dt: numerical integration time step
        :param record: list of observables to record
        :return: dictionary of recorded items
        """

        if record is None:

            record = ['spikes']

        n_steps = len(drives.values()[0])

        results = {}

        for variable in record:

            if variable in ['spikes', 'vs', 'gs']:

                results[variable] = np.nan * np.zeros((n_steps, self.n_neurons))

            elif variable in ['fre_items']:

                results[variable] = np.nan * np.zeros((n_steps, self.n_items))

        rp_ctrs = np.zeros((self.n_neurons,))

        for t_ctr in range(n_steps):

            if t_ctr == 0:

                v = v_init.copy()
                g = g_init.copy()
                fre_items = np.zeros((self.n_items,))

            else:

                # calculate voltage from all inputs

                drive = {key: val[t_ctr] for key, val in drives.items()}

                noise = self.noise * np.random.normal(0, 1, (self.n_neurons,))

                g = self.calc_synaptic_conductances(g=g, spikes=spikes, dt=dt)
                v = self.calc_voltage(v=v, g=g, drive=drive, noise=noise, dt=dt)

            # set neurons in refractory period back to v_reset

            v[rp_ctrs > 0] = self.v_reset

            # perform k-WTA inhibition

            if np.sum(fre_items > self.wta_th) >= self.k_wta:

                # get indices of neurons with largest firing rates

                idxs_winners = np.argpartition(fre_items, -self.k_wta)[-self.k_wta:]

                # build inhibitory input vector

                input_inh = self.wta_inh * np.ones((self.n_items,))
                input_inh[idxs_winners] = 0

                # apply inhibitory input to voltages

                v[self.item_idxs] += input_inh

            # make sure no neurons are below v_min

            v[v < self.v_min] = self.v_min

            # allow neurons that are still above threshold to spike

            spikes = (v > self.v_th).astype(float)
            v[spikes == 1] = self.v_reset

            # update firing rate estimate

            fre_items += (dt / self.tau_fre) * (-fre_items) + (spikes[self.item_idxs] / self.tau_fre)

            # decrement refractory period counters and reset counters for spiking neurons

            rp_ctrs[rp_ctrs > 0] -= dt

            rp_ctrs[spikes == 1] = self.t_rp

            # store data

            if 'spikes' in record:

                results['spikes'][t_ctr, :] = spikes.copy()

            if 'vs' in record:

                results['vs'][t_ctr, :] = v.copy()

            if 'gs' in record:

                results['gs'][t_ctr, :] = g.copy()

            if 'fre_items' in record:

                results['fre_items'][t_ctr, :] = fre_items

        ts = np.arange(n_steps) * dt

        return results, ts