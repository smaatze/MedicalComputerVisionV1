
from dataio.transformation.transforms import Transformations

def get_dataset_transfomation(name, opts=None):

    trans_obj = Transformations(name)
    if opts: trans_obj.initialise(opts)

    # print the input options
    trans_obj.print()

    # return a dictionary of transformations
    return trans_obj.get_transformation()
