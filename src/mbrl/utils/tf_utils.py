import os
import tensorflow as tf

from mbrl.experiments import logger


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def config_gpu(gpu=0, gpu_frac=0.3):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_frac
    return config


def enable_eager_execution(gpu=0, gpu_frac=0.3):
    tf.enable_eager_execution(
        config=config_gpu(gpu=gpu, gpu_frac=gpu_frac),
        # device_policy=tf.contrib.eager.DEVICE_PLACEMENT_WARN
    )


def enable_static_execution(gpu=0, gpu_frac=0.3):
    graph = tf.Graph()
    session = tf.Session(graph=graph, config=config_gpu(gpu=gpu, gpu_frac=gpu_frac))
    session.__enter__() # so get default session works


def restore_checkpoint(model, fname):
    logger.debug('Restoring model {0}'.format(fname))
    assert tf.train.checkpoint_exists(fname)
    checkpointer = tf.train.Checkpoint(model=model)
    status = checkpointer.restore(fname)
    if not tf.executing_eagerly():
        status.initialize_or_restore(tf.get_default_session())
