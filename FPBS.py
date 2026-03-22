'''
===============================================================================
File: FPBS.py
Author: Zoe Fisch (zoefisch@umich.edu), Ella Ricci (earicci@umich.edu), Maria Herrmann (marherr@umich.edu)
Date: 2026-3-17
Group: University of Michigan SunRISE Mission

Description: Frequency-Persistent Background Suppression is the third stage of our preprocessing pipeline. 
By systematically addressing frequency-dependent artifacts while preserving transient burst signals, 
FPBS will improve the quality of data available to downstream detection and classification algorithms, 
ultimately enhancing our ability to study solar radio emissions. 


===============================================================================
'''
import numpy as np

def temporal_median_intensity(S):
    if S.ndim == 0:
        raise ValueError("Input spectrogram must not be empty")
    elif S.ndim != 2:
        raise ValueError("Input spectrogram must be 2D, given spectrogram is ", S.ndim, " dimensions")
    mu = np.median(S, axis=1)
    return mu       

def temporal_variability(S, mu):
    deviation = np.abs(S - mu[:, np.newaxis])
    sigma = 1.4826 * np.median(deviation, axis=1)
    return sigma

# Score P = 1 --> band is persistent
# Score P = 0 --> band is quiet / transient
def persistence_score(S, mu, sigma, k=2.0):
    theta = mu + k * sigma
    P = np.mean(S > theta[:, None], axis=1)
    return P

    



        




# computer persistence scores 

# classify channels 

# build supression mask 

# supress persistent channels 

# reconstruct spectrogram

# evaluation 