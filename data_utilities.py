import pandas as pd
import numpy as np
from os.path import join
from tqdm import trange


_inputfolder = join('data', 'congedo')
_outputfolder = join('data', 'covariances')


def _obtain_path(filename, folder=_inputfolder):
    return join(folder, filename)


def _subjectfile(number, eyes=None):
    if eyes is None:
        return "subject_{:02}.csv".format(number)
    else:
        return "subject_{:02}_cov_{}.csv".format(number, eyes)


def _read_data(filepath, head):
    """
    Reads the full dataset.
    """
    return pd.read_csv(filepath, header=None, names=head).set_index('Time')


def _save_cov(data, subjectnumber, eyes):
    filepath = _obtain_path(_subjectfile(subjectnumber, eyes), _outputfolder)
    l = data.shape[0]
    pd.DataFrame(data=data, columns=range(l), index=range(l)).to_csv(filepath)


def _closed_eyes(df):
    """
    Creates boolean indicator of closed eyes.
    """
    closed = np.zeros(shape=(len(df)), dtype=bool)
    for i, x in enumerate(df.itertuples()):
        if x.EyesClosed:
            closed[i] = True
        elif x.EyesOpened:
            closed[i] = False
        elif closed[i-1]:
            closed[i] = True
    return closed


def _divide_closed(df):
    """
    Divides the dataset in two, one for closed eyes and one for open eyes.
    """
    cl_ind = _closed_eyes(df[['EyesClosed', 'EyesOpened']])
    cl = df[cl_ind].iloc[:, :-3]
    op = df[~cl_ind].iloc[:, :-3]
    return cl, op


def _normalize(df, kind='mean'):
    """
    Normalizes columns according to selected method.

    Available methos:
        - mean: (data - mean) / std
        - minmax: (data - min) / (max - min)
    
    Raise:
        - ValueError if kind is not recognized
    """
    if kind == 'mean':
        return df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    elif kind == 'minmax':
        return df.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
    else:
        raise ValueError("Only mean and minmax normalizations availables")


def create_covariances(subjectnumber, head=None):
    """
    Given a subject number returns the covariance matrices for closed eyes and open eyes
    """
    filepath = _obtain_path(_subjectfile(subjectnumber))
    head = pd.read_csv(_obtain_path('header.csv')).columns if head is None else head

    try:
        df = _read_data(filepath, head)
    except FileNotFoundError:
        raise FileNotFoundError("The subject searched is not available")

    cl, op = _divide_closed(df)

    cl = _normalize(cl).to_numpy()
    op = _normalize(op).to_numpy()

    cl_cov = cl.T @ cl
    op_cov = op.T @ op

    return cl_cov, op_cov


def read_demographics():
    demo_heads = pd.read_csv(_obtain_path('demographic_header.csv')).columns
    demographics = pd.read_csv(_obtain_path('demographic.csv'), header=None, names=demo_heads)
    return demographics




if __name__ == "__main__":

    #### DEMOGRAPHICS
    demographics = read_demographics()

    #### IMPORT DATA HEADER
    head = pd.read_csv(_obtain_path('header.csv')).columns
    l = len(head) - 3

    for i in trange(10):
        subject = i + 1
        
        mats = create_covariances(subject, head)
        
        for j, eyes in enumerate(['closed', 'open']):
            _save_cov(mats[j], subject, eyes)
        


