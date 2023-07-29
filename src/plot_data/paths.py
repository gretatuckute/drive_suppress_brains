import os
from os.path import join
import getpass
user = getpass.getuser()

ROOTDIR = os.path.abspath(join(os.path.dirname( __file__ ), '..'))

# Default paths
RESULTROOT 	= join(ROOTDIR, 'results')
WEIGHTROOT 	= join(ROOTDIR, 'regr-weights' ) # Directory for storing regression weights
DATAROOT	= join(ROOTDIR, 'data')
ACTVROOT    = join(ROOTDIR, 'model-actv') # Directory for storing model activations
LOGROOT     = join(ROOTDIR, 'logs') # Directory for storing logs

# Generate each of these directories if they don't already exist
if not os.path.exists(RESULTROOT):
    os.makedirs(RESULTROOT)
    print(f'Created RESULTROOT: {RESULTROOT}')
if not os.path.exists(WEIGHTROOT):
    os.makedirs(WEIGHTROOT)
    print(f'Created WEIGHTROOT: {WEIGHTROOT}')
if not os.path.exists(DATAROOT):
    os.makedirs(DATAROOT)
    print(f'Created DATAROOT: {DATAROOT}')
if not os.path.exists(ACTVROOT):
    os.makedirs(ACTVROOT)
    print(f'Created ACTVROOT: {ACTVROOT}')
if not os.path.exists(LOGROOT):
    os.makedirs(LOGROOT)
    print(f'Created LOGROOT: {LOGROOT}')
