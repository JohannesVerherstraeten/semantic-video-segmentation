
def get_instance(module, config, *args, **kwargs):
    """
    :param module: module where the instance is defined.
    :type module: Module
    :param config: configuration of the instance.
        config["type"] must contain the name of the instance to be created.
        config["kwargs"] may contain a dictionary of keyword arguments to feed to the constructor of the instance.
    :type config: dict[str, Any]
    :param args: arguments to feed to the constructor of the instance.
    :type args:
    :param kwargs: keyword arguments to feed to the constructor of the instance. Overwrites same-named config kwargs.
    :type kwargs:
    :return: a new instance of type `config["type"]`
    :rtype: Any
    """
    new_kwargs = {**config['kwargs'], **kwargs}     # overwrite config kwargs with given kwargs
    return getattr(module, config['type'])(*args, **new_kwargs)


def get_dataset_instances(module, config, *args, **kwargs):
    """
    :param module: module where the instance is defined.
    :type module: Module
    :param config: configuration of the instance.
        config["type"] must contain the name of the instance to be created.
        config["kwargs"] may contain a dictionary of keyword arguments to feed to the constructor of the instance.
    :type config: dict[str, Any]
    :param args: arguments to feed to the constructor of the instance.
    :type args: Any
    :param kwargs: keyword arguments to feed to the constructor of the instance. Overwrites same-named config kwargs.
    :type kwargs:
    :return: a new training-, validation- and testing instance of type `config["type"]`
    :rtype: (data.dataset.basedataset.BaseDataset, data.dataset.basedataset.BaseDataset, data.dataset.basedataset.BaseDataset)
    """
    new_kwargs = {**config['kwargs'], **kwargs}  # overwrite config kwargs with given kwargs
    return getattr(module, config['type']).create(*args, **new_kwargs)


