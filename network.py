from __future__ import division, print_function
import numpy as np


def make_basic_weight_matrix(n_items, n_assocs, p_cxn, w_ia, w_ai):
    """
    Make a weight matrix corresponding to a set of items
    :param n_items:
    :param n_assocs:
    :param p_cxn:
    :param w_ia: cxn strength from assocs to items
    :param w_ai: cxn strength from items to assocs
    :return:
    """
    n_total = n_items + n_assocs
    w = np.zeros((n_total, n_total))

    w_partial = np.random.rand(n_items, n_assocs) < p_cxn

    w[:n_items, n_items:n_total] = w_ia * w_partial
    w[n_items:n_total, :n_items] = w_ai * w_partial.T

    return w


class Basic(object):
    """
    Basic network of units with activation-triggered hyperexcitability.
    """

    def __init__(self, ths, g_xs, t_xs, n_items, w):

        self.ths = ths
        self.g_xs = g_xs
        self.t_xs = t_xs
        self.n_items = n_items
        self.w = w
        self.n_total = w.shape[0]

    def parse_control(self, control, xc):
        """Compute control inputs"""
        v_control = np.zeros(self.n_total)

        if control is not None:
            type = control[0]

            if type == 'reset': xc = np.zeros(self.n_total)
            elif type == 'item_blanket': v_control[:self.n_items] = control[1]
            elif type == 'assoc_blanket': v_control[self.n_items:] = control[1]
            elif type == 'full_blanket': v_control[:] = control[1]

        return v_control, xc

    def parse_local_input(self, local_input):
        """Compute local inputs"""
        v_local = np.zeros(self.n_total)

        if local_input is not None:
            nodes, inputs = local_input
            v_local[nodes] = inputs

        return v_local

    def run(self, local_inputs, controls):
        """
        Drive the network with a set of local and control inputs.
        :param local_inputs: list of local input summaries
        :param controls: list of control input summaries
        :return: activations, hyperexcitabilities, summed inputs
        """
        assert len(local_inputs) == len(controls)

        rs = []
        xcs = []
        vs = []

        # assume initial condition of everything silent
        r = np.zeros(self.n_total)
        xc = np.zeros(self.n_total)

        for local_input, control in zip(local_inputs, controls):

            # parse external inputs and controls
            v_control, xc = self.parse_control(control, xc=xc)
            v_local = self.parse_local_input(local_input)

            # compute upstream and hyperexcitability inputs
            v_upstream = self.w.dot(r)
            v_x = self.g_xs * (xc > 0).astype(float)

            # compute total inputs and activations
            v = v_control + v_local + v_upstream + v_x
            r = (v >= self.ths).astype(float)

            # decrement old hyperexcitability counters and set new ones
            xc[xc > 0] -= 1
            if type(self.t_xs) in (int, float):
                xc[r > 0] = self.t_xs
            else:
                xc[r > 0] = self.t_xs[r > 0]

            # store everything
            rs.append(r)
            xcs.append(xc.copy())
            vs.append(v_control + v_local + v_upstream)

        rs = np.array(rs)
        xcs = np.array(xcs)
        vs = np.array(vs)

        return rs, xcs, vs
