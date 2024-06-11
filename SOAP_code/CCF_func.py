import numpy as np
# from numba import njit
c = 299792458.

#@njit
def calculate_CCF_1(
        velocity_array, wavelength, wavelength_line, weight_line, 
        wavelength_extend):
    """
    Calculate the Cross-Correlation Function (CCF) templates per velocity.

    Parameters:
        velocity_array (array): Array of velocities at which to calculate the CCF.
        wavelength (array): Array of observed wavelengths.
        wavelength_line (array): Array of reference line wavelengths.
        weight_line (array): Array of weights for each reference line.
        wavelength_extend (array): Extended wavelength array for interpolation.

    Returns:
        array: A 2D array where each row corresponds to the CCF template at a given velocity.
    """
    mask_template = np.zeros((velocity_array.size, wavelength.size))
    index_old_begin =20000
    index_old_end = 20000

    begin_wave = wavelength_extend.min()

    for i in np.arange(velocity_array.size):
        
        # Shift the reference line wavelengths by the current velocity
        wavelength_line_shift = wavelength_line*(1+velocity_array[i]/c)
        
        # Calculate the mask for the current velocity
        mask_corr = calc_mask(wavelength_extend, wavelength_line_shift, weight_line, begin_wave)
        mask_corr = mask_corr[index_old_begin:-index_old_end]

        mask_template[i,:] = mask_corr


    return mask_template

#@njit
def calculate_CCF_2(velocity_array, spectrum, mask_template):

    CCF = np.zeros(velocity_array.size)

    for i in np.arange(velocity_array.size):

        ccf_vel = np.sum(spectrum*mask_template[i,:])
        CCF[i] = ccf_vel

    return CCF

#@njit
def Delta_wavelength(v,wavelength0):
    beta = v/c
    delta_wavelength = wavelength0 * (np.sqrt((1+beta)/(1-beta))-1)
    return delta_wavelength

#@njit
def calc_mask(
        wavelength_extend ,wavelength_line_shift, weight_line, begin_wave, 
        mask_width=820, hole_width=0):


    hole_width = np.array(
        [Delta_wavelength(mask_width, wavelength_line_shift[i]) 
            for i in np.arange(len(wavelength_line_shift))]
            )

    begining_mask_hole = wavelength_line_shift-hole_width/2.
    end_mask_hole = wavelength_line_shift+hole_width/2.

    index_begining_mask_hole = []
    index_end_mask_hole = []

    freq_step_before_mask_hole = []
    freq_step_after_mask_hole = []
    bg_wave = wavelength_extend.min()

    for i in np.arange(len(wavelength_line_shift)):
        # Calculate the indices for the beginning and end of each mask hole
        aa = int(np.ceil((begining_mask_hole[i] - begin_wave)/0.005))
        bb = int(np.ceil((end_mask_hole[i] - begin_wave)/0.005)-1)


        index_begining_mask_hole.append(aa)
        index_end_mask_hole.append(bb)

        # Calculate the frequency step before and after each mask hole
        freq_step_before_mask_hole.append(wavelength_extend[aa] - wavelength_extend[aa-1])
        freq_step_after_mask_hole.append(wavelength_extend[bb+1] - wavelength_extend[bb])


    mask = np.zeros(wavelength_extend.size)
    a = np.array(index_begining_mask_hole)
    b = np.array(index_end_mask_hole)

    freq_step_before_mask_hole = np.array(freq_step_before_mask_hole)
    freq_step_after_mask_hole = np.array(freq_step_after_mask_hole)


    fraction_pixel_before_mask_hole = np.abs(wavelength_extend[a] - begining_mask_hole)/freq_step_before_mask_hole
    fraction_pixel_after_mask_hole  = np.abs(wavelength_extend[b] - end_mask_hole)/freq_step_after_mask_hole

    for i in np.arange(a.size):

        mask[a[i]:b[i]] = [weight_line[i]]*(b[i]-a[i])
        mask[a[i]-1] = weight_line[i]*fraction_pixel_before_mask_hole[i]
        mask[b[i]] = weight_line[i]*fraction_pixel_after_mask_hole[i]

    return mask
