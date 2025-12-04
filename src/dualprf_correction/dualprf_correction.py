import numpy as np
from scipy import ndimage

"""
dualporf_correction
===========
Correct dual-PRF dealiasing errors
"""


def process_pyodim(
    dataset,
    vel_field="velocity",
    kernel_det=np.ones((7, 7)),
    min_valid_det=1,
    max_dev=1.0,
    two_step=True,
    kernel_cor=None,
    min_valid_cor=1,
    new_field_name="velocity_cor",
):
    """
    Correction of dual-PRF outliers in radar velocity data.
    Includes the corrected field in the input radar object.
    Uses technique:
    'cmean' : local circular mean velocity (Hengstebeck et al., 2018)


    Parameters
    ----------
    dataset : list
        list of pyodim dicts
    vel_field: str
        Input velocity field name (dual-PRF)
    kernel_det : array
        Neighbour kernel, 1/0 values (detection), if None a 7x7 ones array
        is used, excluding the central value
    min_valid_det : int
        Minimum number of valid neighbours (detection)
    max_dev : float
        Maximum deviation threshold (detection)
    two_step : bool
        Whether to separate detection and correction stages
    kernel_cor : array
        Neighbour kernel 1/0 values (correction), if None, kernel_det is used
    min_valid_cor : int
        Minimum number of valid neighbours (correction)
    new_field_name : str
        Output (corrected) velocity field name

    """
    n_sweeps = len(dataset) - 1

    for sweep_idx in reversed(list(range(n_sweeps))):
        #print("sweep", sweep_idx, "elevation:", dataset[sweep_idx].elevation.values[0])
        # Skip sweeps without PRT information (single PRF)
        prt = dataset[sweep_idx].prt
        if prt is None:
            #print("  Skipping sweep - not dual-PRF")
            continue

        # Dual-PRF parameters
        v_ny = dataset[sweep_idx].NI
        prf_ratio = np.max(prt) / np.min(prt)
        prf_factor = int(round(1 / (prf_ratio - 1), 0))

        prf_factor_sw = np.zeros_like(prt) + prf_factor
        prf_factor_sw[np.where(prt == np.max(prt))] += 1
        prf_factor_sw = np.transpose(
            np.tile(prf_factor_sw.astype(int), (dataset[sweep_idx].sizes["range"], 1))
        )

        # primary velocities
        vel_primary = v_ny / prf_factor
        # extract velocity sweep
        vel_data = np.ma.masked_invalid(dataset[sweep_idx][vel_field].values)

        # ERROR DETECTION
        # Reference velocities at each gate

        vref_det = _vel_ref(
            data_ma=vel_data,
            kernel=kernel_det,
            v_ny=v_ny,
            mask=None,
            min_valid=min_valid_det,
        )

        # Calculate difference in phase space
        ph_obs = vel_data * (np.pi / v_ny)
        ph_ref = vref_det * (np.pi / v_ny)
        ph_diff = ph_obs - ph_ref
        diff_ma = (v_ny / np.pi) * np.ma.arctan2(np.ma.sin(ph_diff), np.ma.cos(ph_diff))

        # Outlier mask
        err_mask = np.zeros(diff_ma.shape)
        err_mask[np.ma.abs(diff_ma) > (max_dev*vel_primary)] = 1
        err_mask[diff_ma.mask] = 0

        if two_step:
            vref_cor = _vel_ref(
                data_ma=vel_data,
                kernel=kernel_cor,
                v_ny=v_ny,
                mask=err_mask.astype(bool),
                min_valid=min_valid_cor,
            )

        else:
            vref_cor = vref_det

        # ERROR CORRECTION
        # Unwrap number and corrected velocity field
        unwrap_field = _dualprf_error_unwrap(
            data_ma=vel_data,
            ref_ma=vref_cor,
            err_mask=err_mask,
            pvel_arr=vel_primary,
            prf_arr=prf_factor_sw,
        )

        # Correct velocity field
        vel_data_corrected = vel_data + 2 * unwrap_field * vel_primary

        # Fold velocity values into Nyquist interval
        vel_data_corrected_folded = fold_circular(data_ma=vel_data_corrected, mod=v_ny)

        # add field
        dataset[sweep_idx] = dataset[sweep_idx].merge(
            {new_field_name: (("azimuth", "range"), vel_data_corrected_folded)}
        )
    return dataset


def fold_circular(data_ma, mod):
    """
    Values outside the specified interval are folded back into
    the interval.

    Parameters
    ----------
    data_ma : masked array
        Data
    mod: float
        Interval (module)

    Returns
    -------
    ma_fold :  masked array
        Folded data
    """

    # Phase space
    ph = data_ma * np.pi / mod
    ph_fold = np.ma.arctan2(np.ma.sin(ph), np.ma.cos(ph))

    # Back to original variable
    ma_fold = ph_fold * mod / np.pi

    return ma_fold


def local_cmean(data_ma, kernel):
    """
    Calculates local circular mean of a masked array;
    edges are wrapped in azimuth and padded with NA in range.

    Parameters
    ----------
    data_ma : masked array
        Data
    kernel : array
        Local neighbour kernel, 1/0 values

    Returns
    -------
    cmean_ma : masked array
        Local circular mean of the data.
    """

    # Arrays of trigonometric variables
    cos_ma = np.ma.cos(data_ma)
    sin_ma = np.ma.sin(data_ma)

    # Arrays with local means of trigonometric variables
    cos_avg = local_mean(cos_ma.data, cos_ma.mask, kernel)
    sin_avg = local_mean(sin_ma.data, sin_ma.mask, kernel)

    # Local circular mean
    cmean_ma = np.ma.arctan2(sin_avg, cos_avg)

    return cmean_ma


def local_mean(data, mask, kernel):
    """
    Calculates local mean of a masked array;
    edges are wrapped in azimuth and padded with NA in range.

    Parameters
    ----------
    data_ma : masked array
        Data
    kernel : array
        Local neighbour kernel, 1/0 values

    Returns
    -------
    avg_ma : masked array
        Local mean of the data.
    """

    # Local number of valid neighbours

    # pad in range to allow for convolution wrap
    col_num, padded_data = pad_in_range(data * (~mask), kernel, value=0)
    col_num, padded_mask = pad_in_range(mask, kernel, value=0)

    # Replace NaN with 0 and create a weight map
    data_filled = np.nan_to_num(padded_data, nan=0.0)
    weights = np.logical_or(padded_mask, np.isnan(padded_data))

    # Convolve both data and weights
    data_conv = ndimage.convolve(data_filled, kernel, mode="wrap", cval=0.0)
    weights_conv = ndimage.convolve(
        weights.astype(float), kernel, mode="wrap", cval=0.0
    )

    # Normalize by weights
    sum_arr = np.divide(data_conv, weights_conv, 
                   out=np.full_like(data_conv, np.nan),
                   where=weights_conv > 0)

    # Remove added columns
    sum_arr = sum_arr[:, : (sum_arr.shape[1] - col_num)]

    # Calculate average
    avg_ma = np.ma.array(data=sum_arr, mask=mask)

    return avg_ma


def _dualprf_error_unwrap(data_ma, ref_ma, err_mask, pvel_arr, prf_arr):
    """
    Finds the correction factor that minimises the difference between
    the gate velocity and the reference velocity

    Parameters
    ----------
    data_ma : masked array
        Data
    ref_ma : masked array
        Reference data
    err_mask : bool array
        Mask for the identified outliers
    pvel_arr : array
        Primary (high/low PRF) velocity for each gate
    prf_arr : array
        PRF (high/low) of each gate

     Returns
     -------
     nuw : int array
         Unwrap number (correction factor) for each gate
    """

    # Convert non-outliers to zero
    ma_out = data_ma * err_mask.astype(int)
    th_arr_out = pvel_arr * err_mask.astype(int)
    ref_out = ref_ma * err_mask.astype(int)

    # Primary velocity and prf factor of low PRF gates
    prf_factor = np.unique(np.min(prf_arr))[0]
    th_l = th_arr_out.copy()
    th_l[prf_arr == prf_factor] = 0

    dev = np.ma.abs(ma_out - ref_out)
    nuw = np.zeros(ma_out.shape)

    # Loop for possible correction factors
    for ni in range(-prf_factor, (prf_factor + 1)):

        # New velocity values for identified outliers
        if abs(ni) == prf_factor:
            v_corr_tmp = ma_out + 2 * ni * th_l
        else:
            v_corr_tmp = ma_out + 2 * ni * th_arr_out

        # New deviation for new velocity values
        dev_tmp = np.ma.abs(v_corr_tmp - ref_out)
        # Compare with previous deviation
        delta = dev - dev_tmp

        # Update unwrap number when deviation has decreased
        nuw[delta > 0] = ni
        # Update corrected velocity and deviation
        v_corr = ma_out + 2 * nuw * th_arr_out
        dev = np.ma.abs(v_corr - ref_out)

    return nuw.astype(int)


def pad_in_range(data, kernel=np.ones((3, 3)), value=None):
    """
    Add dummy (e.g. NA/NAN) values in range so that 'wrap' property can
    be applied in convolution operations.

    Parameters
    ----------
    data : array
        Data
    kernel : array
        Neighbour kernel 1/0 values
    value : float or None
        Value set in dummy columns

    Returns
    -------
    col_num : int
        Number of columns added.
    data_out : array
        Data with added dummy columns.
    """

    c = (np.asarray(kernel.shape) - 1) / 2  # 'center' of kernel
    col_num = int(np.ceil(c[1]))

    cols = np.zeros((data.shape[0], col_num))

    if value is None:
        cols[:] = np.nan
    else:
        cols[:] = value

    # Add dummy columns
    data_out = np.hstack((data, cols))

    return col_num, data_out

def _vel_ref(data_ma, kernel=np.ones((5, 5)), v_ny=None, mask=None, min_valid=1):
    """
    Estimate reference velocity using different local statistics:
    'cmean' : local circular mean velocity (Hengstebeck et al., 2018)

    Parameters
    ----------
    data_ma : masked array
        Data
    kernel : array
        Neighbour kernel (1/0 values)
    v_ny : float
        Nyquist velocity
    mask : bool array
        User-defined mask
    min_valid : int
        Minimum number of valid neighbours

     Returns
     -------
     v_ref : array
         Reference velocity for each gate
    """

    if mask is None:
        mask = data_ma.mask
    else:
        mask = np.ma.mask_or(data_ma.mask, mask)

    vel_ma = np.ma.array(data=data_ma.data, mask=mask)

    # Mask gates which do not have a minimum number of neighbours
    valid_num_arr = ndimage.convolve((~mask).astype(int), kernel, mode="constant", cval=0)
    nmin_mask = np.zeros(mask.shape)
    nmin_mask[valid_num_arr < min_valid] = 1
    
    new_mask = np.ma.mask_or(data_ma.mask, nmin_mask.astype(bool))

    ph_arr = vel_ma * (np.pi / v_ny)

    v_ref = (v_ny / np.pi) * local_cmean(ph_arr, kernel=kernel)

    v_ref = np.ma.array(data=v_ref.data, mask=new_mask)
    v_ref = fold_circular(v_ref, mod=v_ny)

    return v_ref