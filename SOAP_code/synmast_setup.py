import numpy as np
import matplotlib.pyplot as plt
import bz2
import pandas as pd
from scipy.interpolate import interp1d
import configparser
config = configparser.ConfigParser()
import os, sys
from pathlib import Path
from astropy.io import fits

from rdmag import rdmag_synmast
from rebin_jv import rebin_jv

# Add the directory to sys.path
synmast_path = '/home/st79/Github_mycode/pySYNTHMAG'
sys.path.append(synmast_path)

from pySYNTHMAG import run


def vac2air(wavelength):
    '''
    # Transform wavelength in vacuum to air
    # See VALD website here
    # http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    # The formula comes from Birch and Downs (1994, Metrologia, 31, 315)
    '''
    s = 1.e4 / wavelength # s = 10^4 / lamda_vac, lamda_vac is in Angstrom
    n_air = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
    return wavelength / n_air


def air2vac(wavelength):
    '''
    # Transform wavelength in air to vacuum
    # See VALD website here
    # http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    # The formula comes from N. Piskunov
    '''
    s = 1.e4 / wavelength # s = 10^4 / lamda_vac, lamda_vac is in Angstrom
    n_air = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 0.0001599740894897 / (38.92568793293 - s**2)
    return wavelength * n_air


def PHOENIX_name(temp, logg_val, FeH_val, alpha_val):

    if FeH_val <= 0.0:
        prefix_subdir = 'Z-%.1f' % np.abs(FeH_val)
        if alpha_val == 0.0:

            name = '/lte0' + str(int(temp)) + '-%.2f-%.1f.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits' % (logg_val, np.abs(FeH_val))
        elif alpha_val > 0.0:

            name = '/lte0' + str(int(temp)) + '-%.2f-%.1f.Alpha=+%.2f.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits' % (logg_val, np.abs(FeH_val), alpha_val)
        else:

            name = '/lte0' + str(int(temp)) + '-%.2f-%.1f.Alpha=%.2f.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits' % (logg_val, np.abs(FeH_val), alpha_val)

    else:
        prefix_subdir = 'Z+%.1f' % np.abs(FeH_val)
        name = '/lte0' + str(int(temp)) + '-%.2f+%.1f.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits' % (logg_val, np.abs(FeH_val))

    return prefix_subdir, name


def mask_templates(Teff_star):

    mask_type = np.array(
        ['F9', 'G2', 'G8', 'G9', 'K2', 'K5', 'K6', 'M0', 'M2', 'M3', 'M4', 'M5'])
    Teff_template_array = np.array(
        [6050, 5770, 5480, 5380, 5100, 4400, 4300, 3800, 3400, 3250, 3100, 2800])

    index = np.argmin(np.abs(Teff_template_array - Teff_star))
    mask_template_name = mask_type[index]+'_mask.txt'

    return mask_template_name



if __name__ == '__main__':

    # --- Setup using the config file ---

    config_name = 'config_conti_phoenix.cfg'
    config.read(config_name)

    grav = float(config.get('star','logg' ))
    wvmin = float(config.get('star','minwave' ))
    wvmax = float(config.get('star','maxwave' ))
    FeH = 0.0
    alpha = 0.0

    input_prefix = config.get('data_io','input_prefix')
    out_path = input_prefix+config.get('data_io','input_dir_source')

    rassine_path = config.get('data_io','Rassine_directory')


    # --- Get the Phoenix data with the closest parameters ---

    logg_array = np.linspace(0.0,6.0,13)
    temp_array = np.zeros(73)
    FeH_array = np.concatenate( (np.linspace(-4.0, -3.0, 2), np.linspace(-2.0, 1.0, 7)) )
    FeH_array = FeH_array.round(decimals=2)
    alpha_array = -0.2+np.arange(8)*0.2
    alpha_array = alpha_array.round(decimals=2)

    for j in np.arange(73):
        if j < 48:
            temp_array[j] = 2300.0 + 100.0*(j)
        else:
            temp_array[j] = 7000.0 + 200.0*(j-49)

    # print(temp_array.min())
    index_logg = np.argmin(np.abs(logg_array - grav))
    index_FeH = np.argmin(np.abs(FeH_array - FeH))
    index_alpha = np.argmin(np.abs(alpha_array - alpha))


    # --- Download the Phoenix data ---

    prefix_phoenix = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/'
    prefix_phoenix_wv = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/'

    # mask_type = np.array(['F9', 'G2', 'G8', 'G9', 'K2',])
    # Teff_array = np.array([6050, 5778, 5480, 5380, 5100])

    Temp  = int(config.get('star','Tstar'))
    Teff_array = np.array([Temp])

    for teff_id in np.arange(Teff_array.size):

        tstar  = Teff_array[teff_id]
        tspot  = tstar - int(config.get('star','Tdiff_spot' ))
        tfaculae = tstar + int(config.get('star','Tdiff_faculae' ))

        T_array = np.array([tstar, tspot, tfaculae])

        for t_id in np.arange(T_array.size):
            temp = T_array[t_id]
            print('\n;;;;;;;;;;;;;;;;;;')
            print(f'Downloading {temp}')

            logg_val = logg_array[index_logg]
            FeH_val = FeH_array[index_FeH]
            
            alpha_val = 0.0 if FeH_val > 0.0 else alpha_array[index_alpha]

            index_temp = np.argmin(np.abs(temp_array - temp))
            # Set the temperature range
            if temp > temp_array[index_temp]:
                temp_low = temp_array[index_temp]
                temp_high = temp_array[index_temp+1]
            else:
                temp_low = temp_array[index_temp-1]
                temp_high = temp_array[index_temp]

            wave_name = 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'

            sub_dir_low, name_low = PHOENIX_name(temp_low, logg_val, FeH_val, alpha_val)
            sub_dir_high, name_high = PHOENIX_name(temp_high, logg_val, FeH_val, alpha_val)
            wave_file = Path(out_path+wave_name)
            name_low_file = Path(out_path+name_low)
            name_high_file = Path(out_path+name_high)

            if not wave_file.is_file():
                print('Start to download wavelength data...')
                os.system('curl '+prefix_phoenix_wv+wave_name+' -o '+out_path+wave_name)
            if not name_low_file.is_file():
                print('Start to download spectral data...')
                os.system('curl '+prefix_phoenix+sub_dir_low+name_low+' -o '+out_path+name_low)
            if not name_high_file.is_file():
                print('Start to download spectral data...')
                os.system('curl '+prefix_phoenix+sub_dir_high+name_high+' -o '+out_path+name_high)

            wave_data_vac = fits.getdata(out_path+wave_name)
            # Extend the wavelength range by 500 Angstrom on each side
            waveIndex = (wave_data_vac >= (wvmin-500)) & (wave_data_vac <= (wvmax+500))
            wavelength_vac = wave_data_vac[waveIndex]

            spec_low = fits.getdata(out_path+name_low)[waveIndex]
            spec_high = fits.getdata(out_path+name_high)[waveIndex]

            # Convert the wavelength from vac to air
            wavelength_air = vac2air(wavelength_vac)
            waveIndex_new = (wavelength_air >= wvmin) & (wavelength_air <= wvmax)
            wavelength_air_new = wavelength_air[waveIndex_new]
            spec_low_air = spec_low[waveIndex_new]
            spec_high_air = spec_high[waveIndex_new]
            flux_temp = np.zeros(wavelength_air_new.size)

            # Linear interpolation of the spectrum between the two closest temperatures
            for i in np.arange(wavelength_air_new.size):
                flux_temp[i] = spec_low_air[i] \
                    + ( (temp - temp_low)/(temp_high - temp_low) )*(spec_high_air[i] - spec_low_air[i])

            # Save the spectrum in a pickle file
            data_name = out_path+'New_Phoenix_'+str(int(temp))+'.p'
            spec_df = pd.DataFrame({'wave':wavelength_air_new,'flux':flux_temp})
            spec_df.to_pickle(data_name)

            cmds = 'python3 '+rassine_path+'Rassine.py '+out_path+'New_Phoenix_'+str(int(temp))+'.p '+out_path+' '+rassine_path
            print(cmds)
            os.system(cmds)
            
            print('Done!')

    print('\n\n Part One Done!')
    print('!!! REMINDER: SPECTRA WAVELENGTH ARE IN AIR !!!\n\n')




    #####################################################################
    # --- Setup Synmast spectra ---
    print('Start Generating Synmast spectra...')

    input_prefix = config.get('data_io','input_prefix')
    input_dir = input_prefix+config.get('data_io','input_dir_source')
    wave_array_name = config.get('data_io','wave_array_name')
    BIS_dir = input_prefix+config.get('data_io','BIS_prefix')
    CCF_windows= float(config.get('data_io','CCF_window'   ))
    CCF_size = float(config.get('data_io','CCF_size'   ))

    output_dir = input_prefix+config.get('data_io','output_dir')
    wave_data = output_dir+wave_array_name
    wavelength = np.fromfile(wave_data,dtype='double')

    vector_size = int(wavelength.size)
    step_size = round(wavelength[3]-wavelength[2],5)
    wave_extend = 100.0

    depth = np.linspace(0.0,1.0,100)
    coeff_frame = np.load(BIS_dir+'coeff_mu_v1.npz',allow_pickle=True)
    mu_coeff = coeff_frame['coeff_obs'][::-1]
    mu_array = coeff_frame['mus'][::-1]

    tstar  = int(config.get('star','Tstar' ))
    tspot  = tstar - int(config.get('star','Tdiff_spot' ))
    tfaculae = tstar + int(config.get('star','Tdiff_faculae' ))
    T_eff  = np.array([tstar, tspot, tfaculae])

    mask_template = input_prefix+config.get('data_io','mask_prefix') \
        + mask_templates(tstar)
    templates = np.loadtxt(mask_template)
    freq_line = templates[:,0]
    contrast_line = templates[:,1]
    index_lines = (freq_line > wavelength.min()) & (freq_line < wavelength.max()) & (contrast_line > 0.1)
    freq_line = freq_line[index_lines]
    contrast_line =contrast_line[index_lines]

    # scale1= cubic_func(tstar, coeff_slope, coeff_offset)/np.abs((vel_interpo_mu8[index]-vel_interpo_mu2[index]))
    # colors = plt.cm.jet(np.linspace(1,0,mu_array.size))

    # no_points = int(wave_extend/step_size)
    # wavelength_before = np.linspace(np.min(wavelength)-step_size*no_points,np.min(wavelength), no_points)
    # wavelength_after = np.linspace(wavelength[-1], wavelength[-1]+step_size*no_points, no_points)
    # wavelength_extend = np.concatenate((wavelength_before,wavelength,wavelength_after))
    # mask_template = calculate_CCF_1(vrad_ccf2,wavelength,freq_line,contrast_line, wavelength_extend)
    
    factor_rescale = 10**15
    
    spectral_type = np.array(['quiet', 'spot', 'faculae'])
    for i_t in np.arange(T_eff.size):

        print(i_t, T_eff[i_t])

        # Get Synthmag Spectrum
        prf_dir = run(int(T_eff[i_t]), grav, 0.0, model='btnextgen',
            atm='btnextgen', water='BT')
        
        spec_wave, spec_seed = rdmag_synmast(prf_dir, 0, vrt=2.0, nx=2*5001)
        spec_seed = spec_seed/np.percentile(spec_seed, 99.5)

        # fig, ax = plt.subplots(1, 1, figsize=(5, 2), dpi=300)
        # ax.plot(spec_wave, spec_seed, label='noconti')
        # plt.show()

        spec_seed = rebin_jv(spec_wave, spec_seed, wavelength, False)

        # fig, ax = plt.subplots(1, 1, figsize=(5, 2), dpi=300)
        # ax.plot(wavelength, spec_seed, label='noconti')
        # plt.show()

        # Get PHOENIX spectrum for SED level
        phoenix_name = input_dir+'/RASSINE_New_Phoenix_'+str(int(T_eff[i_t]))+'.p'
        phoenix_frame = pd.read_pickle(phoenix_name)
        phoenix_contin = phoenix_frame['output']['continuum_cubic']
        phoenix_wave = phoenix_frame['wave']

        phoenix_func_contin = interp1d(phoenix_wave, phoenix_contin)
        phoenix_seed_contin = phoenix_func_contin(wavelength)/factor_rescale

        # ---------------------------------
        # coeff_lines = Line_removal(wavelength, spec_seed)
        # vel_inter = np.poly1d(coeff_lines)(depth)

        # CCF_quiet_Sun = calculate_CCF_2(vrad_ccf2, spec_seed, mask_template)
        # CCF_quiet_Sun /= np.max(CCF_quiet_Sun)
        # CCF_quiet_Sun  = 1-((-CCF_quiet_Sun+1)*appod)
        # bis_c, depth_c = bisector_measurement(vrad_ccf2, CCF_quiet_Sun)
        # depth_array_fit = np.linspace(0,1.0,100)

        spec_strong = spec_seed.copy()

        # for iter in np.arange(1):

        #     index_cp = (depth < 0.85) & (depth > depth_c.min())
        #     vel_off = np.mean(bis_c[(depth_c < 0.85) & (depth_c > depth_c.min())]) - np.mean(vel_inter[index_cp])

        #     coeff_lines[-1] = coeff_lines[-1] +vel_off
        #     flux_new = BIS_removal_spectral(wavelength, spec_strong, coeff_lines, 0, 1)

        #     CCF_quiet_Sun = calculate_CCF_2(vrad_ccf2, flux_new, mask_template)
        #     CCF_quiet_Sun /= np.max(CCF_quiet_Sun)
        #     CCF_quiet_Sun  = 1-((-CCF_quiet_Sun+1)*appod)

        #     bis_c1, depth_c1 = bisector_measurement(vrad_ccf2, CCF_quiet_Sun)

        #     bis_cut =  0.8
        #     span_c1 = bis_c1[depth_c1 < bis_cut].max()-bis_c1[depth_c1 < bis_cut].min()
        #     span_c = bis_c[depth_c < bis_cut].max() - bis_c[depth_c< bis_cut].min()

        #     spec_strong = flux_new.copy()


        # if (np.abs(T_eff[i_t]-tstar) <= 0.1):

        seed_cube = np.zeros((mu_array.size,vector_size))
        seed_cube_contin = np.zeros((mu_array.size,vector_size))
        seed_cube_contin_conv = np.zeros((mu_array.size,vector_size))

        for ids, mus in enumerate(mu_array):
            # print(ids,mus)
            # flux_inj = BIS_inject_spectral_quiet(wavelength, flux_new, mu_coeff[ids], scale1)
            flux_inj = spec_strong

            # CCF_quiet_Sun = calculate_CCF_2(vrad_ccf2, flux_inj, mask_template)
            # CCF_quiet_Sun /= np.max(CCF_quiet_Sun)
            # CCF_quiet_Sun  = 1-((-CCF_quiet_Sun+1)*appod)

            seed_cube[ids,:] = flux_inj[:vector_size]
            seed_cube_contin_conv[ids,:] = flux_inj[:vector_size]  *  phoenix_seed_contin[:vector_size]
            seed_cube_contin[ids,:] = phoenix_seed_contin[:vector_size]

            # bis_c1, depth_c1 = bisector_measurement(vrad_ccf2, CCF_quiet_Sun)
            # vel_interpo = np.poly1d(mu_coeff[ids])(depth_c1)

        # out_name1 = output_dir+'/new_convec_obs_'+spectral_type[i_t]+'_synmast.bin'
        out_name1 = output_dir+'/new_noconti_convec_T_eff_'+str(int(T_eff[i_t]))+'.bin'
        seed_cube.tofile(out_name1)
        
        # out_name3 = output_dir+'/new_contin_convec_obs_'+spectral_type[i_t]+'.bin'
        out_name2 = output_dir+'/new_conti_convec_T_eff_'+str(int(T_eff[i_t]))+'.bin'
        seed_cube_contin_conv.tofile(out_name2)


        # out_name2 = output_dir+'/new_contin_obs_'+spectral_type[i_t]+'.bin'
        out_name3 = output_dir+'/new_conti_T_eff_'+str(int(T_eff[i_t]))+'.bin'
        seed_cube_contin.tofile(out_name3)


        # else:
        #     seed_cube = np.zeros((mu_array.size,vector_size))
        #     seed_cube_contin = np.zeros((mu_array.size,vector_size))
        #     seed_cube_contin_conv = np.zeros((mu_array.size,vector_size))


        #     print('Inject BIS difference')


        #     for ids, mus in enumerate(mu_array):
        #         print(ids,mus)

        #         index_min_fac = np.argmin(np.abs(mu_array_faculae - mus) )
        #         print('use '+str(mu_array_faculae[index_min_fac]))

        #         flux_inj = BIS_inject_spectral_active(wavelength, flux_new, coeff_up_fac_array[index_min_fac,:], coeff_low_fac_array[index_min_fac,:], \
        #                                 dep_cut_fac_array[index_min_fac], velo_cut_fac_array[index_min_fac],velo_shift_fac_array[index_min_fac], scale1)


        #         seed_cube[ids,:] = flux_inj[:vector_size]
        #         seed_cube_contin_conv[ids,:] = flux_inj[:vector_size]  *  phoenix_seed_contin[:vector_size]
        #         seed_cube_contin[ids,:] = phoenix_seed_contin[:vector_size]

        #         CCF_quiet_Sun = calculate_CCF_2(vrad_ccf2, flux_inj, mask_template)
        #         CCF_quiet_Sun /= np.max(CCF_quiet_Sun)
        #         CCF_quiet_Sun  = 1-((-CCF_quiet_Sun+1)*appod)


        #         bis_c1, depth_c1 = bisector_measurement(vrad_ccf2, CCF_quiet_Sun)
        #         vel_interpo = np.poly1d(mu_coeff[-1])(depth_c1)


        #     out_name1 = output_dir+'/new_convec_obs_'+spectral_type[i_t]+'.bin'
        #     seed_cube.tofile(out_name1)

        #     out_name2 = output_dir+'/new_contin_obs_'+spectral_type[i_t]+'.bin'
        #     seed_cube_contin.tofile(out_name2)

        #     out_name3 = output_dir+'/new_contin_convec_obs_'+spectral_type[i_t]+'.bin'
        #     seed_cube_contin_conv.tofile(out_name3)



    # os.system('rm -r ccf*bin')
    # os.system('rm -r CCF*txt')



