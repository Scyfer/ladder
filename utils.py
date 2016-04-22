import logging
import os

import numpy as np
import theano
from blocks.extensions import Printing, SimpleExtension
from blocks.main_loop import MainLoop
from blocks.roles import add_role
from pandas import DataFrame, read_hdf

import fuel.streams
import fuel.transformers
from fuel.schemes import BalancedSamplingScheme

logger = logging.getLogger('main.utils')


def shared_param(init, name, cast_float32, role, **kwargs):
    if cast_float32:
        v = np.float32(init)
    p = theano.shared(v, name=name, **kwargs)
    add_role(p, role)
    return p


class AttributeDict(dict):
    __getattr__ = dict.__getitem__

    def __setattr__(self, a, b):
        self.__setitem__(a, b)


class DummyLoop(MainLoop):
    def __init__(self, extensions):
        return super(DummyLoop, self).__init__(algorithm=None,
                                               data_stream=None,
                                               extensions=extensions)

    def run(self):
        for extension in self.extensions:
            extension.main_loop = self
        self._run_extensions('before_training')
        self._run_extensions('after_training')


class ShortPrinting(Printing):
    def __init__(self, to_print, use_log=True, **kwargs):
        self.to_print = to_print
        self.use_log = use_log
        super(ShortPrinting, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        log = self.main_loop.log

        # Iteration
        msg = "e {}, i {}:".format(
            log.status['epochs_done'],
            log.status['iterations_done'])

        # Requested channels
        items = []
        for k, vars in self.to_print.iteritems():
            for shortname, vars in vars.iteritems():
                if vars is None:
                    continue
                if type(vars) is not list:
                    vars = [vars]

                s = ""
                for var in vars:
                    try:
                        name = k + '_' + var.name
                        val = log.current_row[name]
                    except:
                        continue
                    try:
                        s += ' ' + ' '.join(["%.3g" % v for v in val])
                    except:
                        s += " %.3g" % val
                if s != "":
                    items += [shortname + s]
        msg = msg + ", ".join(items)
        if self.use_log:
            logger.info(msg)
        else:
            print msg


class SaveParams(SimpleExtension):
    """Finishes the training process when triggered."""
    def __init__(self, trigger_var, params, save_path, **kwargs):
        super(SaveParams, self).__init__(**kwargs)
        if trigger_var is None:
            self.var_name = None
        else:
            self.var_name = trigger_var[0] + '_' + trigger_var[1].name
        self.save_path = save_path
        self.params = params
        self.to_save = {}
        self.best_value = None
        self.add_condition(('after_training', 'on_interrupt'), self.save)

    def save(self, which_callback, *args):
        if self.var_name is None:
            self.to_save = {v.name: v.get_value() for v in self.params}
        path = self.save_path + '/trained_params'
        logger.info('Saving to %s' % path)
        np.savez_compressed(path, **self.to_save)

    def do(self, which_callback, *args):
        if self.var_name is None:
            return
        val = self.main_loop.log.current_row[self.var_name]
        if self.best_value is None or val < self.best_value:
            self.best_value = val
        self.to_save = {v.name: v.get_value() for v in self.params}


class SaveExpParams(SimpleExtension):
    def __init__(self, experiment_params, dir, **kwargs):
        super(SaveExpParams, self).__init__(**kwargs)
        self.dir = dir
        self.experiment_params = experiment_params

    def do(self, which_callback, *args):
        df = DataFrame.from_dict(self.experiment_params, orient='index')
        df.to_hdf(os.path.join(self.dir, 'params'), 'params', mode='w',
                  complevel=5, complib='blosc')


class SaveLog(SimpleExtension):
    def __init__(self, dir, show=None, **kwargs):
        super(SaveLog, self).__init__(**kwargs)
        self.dir = dir
        self.show = show if show is not None else []

    def do(self, which_callback, *args):
        df = self.main_loop.log.to_dataframe()
        df.to_hdf(os.path.join(self.dir, 'log'), 'log', mode='w',
                  complevel=5, complib='blosc')


class RebalanceUnlabeledDataStream(SimpleExtension):
    """
    A simple extension that rebalances the data_stream for unlabeled data.

    While it is a known fact that class imbalance in a supervised learning
    task negatively affects the performance of the model (unless
    countermeasures are taken), it is as of yet unclear how class imbalance
    in _unlabeled examples_ affects a laddernet's performance.
      Early experiments on MNIST indicate that while it is not catastrophic to
    have some class imbalance, it does signifcantly impact the validation
    performance on labeled data.
      This SimpleExtension aims to mitigate this by "balancing" the class
    distribution on every epoch. That is, we use the current state of the
    network to generate predictions for all unlabeled examples, assume that
    the resulting predictions are correct, and use simple over- and under-
    sampling to ensure that the _predicted class_ distribution is balanced.

    Parameters
    ----------
    original_data_stream: fuel.streams.DataStream
        Original datastream.
    target_tensor: theano.tensor.TensorVariable
        Tensor that acts as 'target' variable, i.e., we will base our
        rebalancing step on the contents of this tensor.
    """
    def __init__(self, predict_extension, target_tensor,
                 dataset, i_unlabeled, batch_size, whiten, cnorm, **kwargs):
        kwargs.setdefault('after_epoch', True)
        kwargs.setdefault('before_first_epoch', False)
        super(RebalanceUnlabeledDataStream, self).__init__(**kwargs)

        # filter out unused source
        self.predict_extension = predict_extension
        self.dataset = dataset
        self.batch_size = batch_size
        self.i_unlabeled = i_unlabeled
        self.whiten = whiten
        self.cnorm = cnorm

        if isinstance(target_tensor, list):
            # assume entry 0 is the actual target
            self.target_tensor = target_tensor[0]
        else:
            self.target_tensor = target_tensor

    def do(self, *args, **kwargs):
        # generate predictions on currently used data_stream
        predicted_unlabeled_targets = np.argmax(
            self.predict_extension.predictions[self.target_tensor.name],
            axis=1)

        print "Class distribution unlabeled predictions: ", \
            np.sum(self.predict_extension.predictions[self.target_tensor.name],
                   axis=0)

        balanced_scheme = \
            BalancedSamplingScheme(targets=predicted_unlabeled_targets,
                                   examples=self.i_unlabeled,
                                   batch_size=self.batch_size)
        from run import Whitening
        balanced_stream = Whitening(
            fuel.streams.DataStream(self.dataset),
            iteration_scheme=balanced_scheme,
            whiten=self.whiten, cnorm=self.cnorm)
        balanced_stream.sources = ('features_unlabeled',)

        # TODO we're abusing the fact that main_loop is accessible from here
        self.main_loop.data_stream.ds_unlabeled = balanced_stream


def prepare_dir(save_to, results_dir='results'):
    base = os.path.join(results_dir, save_to)
    i = 0

    while True:
        name = base + str(i)
        try:
            os.makedirs(name)
            break
        except:
            i += 1

    return name


def load_df(dirpath, filename, varname=None):
    varname = filename if varname is None else varname
    fn = os.path.join(dirpath, filename)
    return read_hdf(fn, varname)


def filter_funcs_prefix(d, pfx):
    pfx = 'cmd_'
    fp = lambda x: x.find(pfx)
    return {n[fp(n) + len(pfx):]: v for n, v in d.iteritems() if fp(n) >= 0}
