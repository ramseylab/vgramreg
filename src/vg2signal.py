import pandas
import scipy.interpolate
import typing
import argparse
import numpy
import skfda.misc.hat_matrix as skfda_hm
import skfda.preprocessing.smoothing as skfda_smoothing
import skfda
import csaps
import matplotlib.pyplot as plt
import numdifftools
from sklearn import metrics

import numpy as np


def get_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(description="vg2signal.py: process " +
                                         "a voltammogram into an analyte " +
                                         " peak signal value")
    arg_parser.add_argument('--log', dest='log', action='store_true',
                            default=False)
    arg_parser.add_argument('--bw', type=float, default=0.02,
                            help="kernel smoothing bandwidth (V)")
    arg_parser.add_argument('--smooth', type=float, default=0.0000001,
                            help="smoothed spline stiffness " +
                            "parameter (bigger is smoother)")
    arg_parser.add_argument('--vcenter', type=float, default=1.073649114,
                            help="specify the analyte peak voltage (V)")
    arg_parser.add_argument('--vwidth', type=float, default=0.135,
                            help="specify the width of the analyte peak (V)")
    arg_parser.add_argument('--recenter', action='store_true',
                             dest='recenter',
                             help="recenter the window on the empirical " +
                             "peak and then re-analyze the data with the " +
                             "new window")
    arg_parser.add_argument('--plot', action='store_true', dest='plot',
                            help='set to true in order to plot the detilted ' +
                            'voltammogram')
    arg_parser.add_argument('filename')
    return arg_parser.parse_args()


def get_num_header_lines(file_obj: typing.TextIO) -> int:
    line_ctr = 0
    ret_ctr = None
    for line in file_obj:
        line_ctr += 1
        if line.startswith("Potential/V"):
            ret_ctr = line_ctr
    file_obj.seek(0)
    return ret_ctr


def read_raw_vg_as_df(filename: str) -> pandas.DataFrame:
    with open(filename, "r") as input_file:
        header_nlines = get_num_header_lines(input_file)
# a single chain of method calls can produce the desired
# two-column dataframe, with negative current in the "I"
# column and with the voltage in the "V" column
        return pandas.read_csv(
            input_file,
            sep=", ",
            engine="python",
            skiprows=header_nlines - 1
        ).drop(
            columns=["For(i/A)", "Rev(i/A)"]
        ).rename(
            columns={"Potential/V": "V",
                     "Diff(i/A)": "I"}
        ).apply(
            lambda r: [r[0], -1E+6 * r[1]],
            axis=1,
            raw=True)

def make_shoulder_getter(vstart: float,
                         vend: float) -> typing.Callable:
    def shoulder_getter_func(v: numpy.array,
                             lisd: numpy.array):
        v_in = numpy.logical_and(v >= vstart, v <= vend)
        spline_model = scipy.interpolate.UnivariateSpline(v[v_in],
                                                          lisd[v_in],
                                                          s=0,
                                                          k=4)

        v_peak = None
        # we are looking for a local minimum of the third derivative between
        # vstart and vend
        spl_mdl_dd = spline_model.derivative(n=2)
        spl_mdl_dd_pred = spl_mdl_dd(v[v_in])

        spl_mdl_ddd = spline_model.derivative(n=3)
        spl_mdl_ddd_pred = spl_mdl_ddd(v[v_in])
        spl_mdl_ddd_b = scipy.interpolate.splrep(v[v_in],
                                                 spl_mdl_ddd_pred)
        spl_mdl_ddd_ppoly = scipy.interpolate.PPoly.from_spline(spl_mdl_ddd_b)
        roots_ddd = spl_mdl_ddd_ppoly.roots(extrapolate=False)
        if len(roots_ddd) == 1:
            v_peak = float(roots_ddd[0])
        elif len(roots_ddd) > 1:
            minsecond = min(spl_mdl_dd_pred)
            idx = (numpy.abs(spl_mdl_dd_pred - minsecond)).argmin()
            vin = list(v[v_in])
            v_peak = vin[idx]
        else:
            minsecond = min(spl_mdl_dd_pred)
            idx = (numpy.abs(spl_mdl_dd_pred - minsecond)).argmin()
            vin = list(v[v_in])
            v_peak = vin[idx]
            print("WARNING: no roots found")
        return (None, v_peak)
    return shoulder_getter_func


def make_smoother(smoothing_bw: float) -> typing.Callable:
    kernel_estimator = skfda_hm.NadarayaWatsonHatMatrix(bandwidth=smoothing_bw)
    kernel_smoother = skfda_smoothing.KernelSmoother(kernel_estimator)

    def smoother_func(x: numpy.array,
                      y: numpy.array) -> numpy.array:
        fd = skfda.FDataGrid(data_matrix=y,
                             grid_points=x)
        res = kernel_smoother.fit_transform(fd).data_matrix.flatten()
        return res

    return smoother_func


def make_signal_getter(vstart: float,
                       vend: float) -> typing.Callable:
    def signal_getter_func(v: numpy.array,
                           lisd: numpy.array):
        v_in = numpy.logical_and(v >= vstart, v <= vend)
        spline_model = scipy.interpolate.UnivariateSpline(v[v_in],
                                                          lisd[v_in],
                                                          s=0,
                                                          k=4)
        spline_model_d = spline_model.derivative(n=1)
        spline_model_d_ppoly = scipy.interpolate.splrep(v[v_in],
                                                        list(map(spline_model_d,
                                                                 v[v_in])), k=4)
        roots_d = scipy.interpolate.PPoly.from_spline(spline_model_d_ppoly).roots(extrapolate=False)
        spline_model_dd = numdifftools.Derivative(spline_model, n=2)
        dd_at_roots = numpy.array(list(map(spline_model_dd, roots_d)))
        critical_point_v = None
        if len(dd_at_roots) > 0:
            ind_peak = numpy.argmin(dd_at_roots)
            if dd_at_roots[ind_peak] < 0:
                critical_point_v = roots_d[ind_peak]
        signal = None
        if critical_point_v is not None:
            signal = -dd_at_roots[ind_peak]
        return (signal, critical_point_v)
    return signal_getter_func


# stiffness: R-style stiffness parameter (non-negative)
def make_detilter(vstart: float,
                  vend: float,
                  stiffness: float) -> typing.Callable:
    assert stiffness >= 0.0, \
        "invalid stiffness parameter (should be " + \
        f"greater than zero): {stiffness}"

    def detilter_func(v: numpy.array, lis: numpy.array):
        v_out = numpy.logical_or(v < vstart, v > vend)
        lis_bg = csaps.csaps(v[v_out], lis[v_out], v,
                             smooth=(1.0 / (1.0 + stiffness)))
        return lis - lis_bg

    return detilter_func


def v2signal(vg_filename: str,
             do_log: bool,
             smoothing_bw: float,
             vcenter: float,
             vwidth: float,
             stiffness: float):

    vg_df = read_raw_vg_as_df(vg_filename)

    if (vg_df['I'].to_numpy() < 0).any():
        return None, None, vg_df, None, None

    if do_log:
        cur_var_name = "logI"
        #vg_df[cur_var_name] = numpy.emath.logn(logbase, vg_df["I"])
        vg_df[cur_var_name] = numpy.log2(vg_df["I"])
    else:
        cur_var_name = "I"

    smoother = make_smoother(smoothing_bw)


    vg_df["smoothed"] = smoother(vg_df["V"], vg_df[cur_var_name].to_numpy())

    shoulder_getter = make_shoulder_getter(1, 1.1)
    (peak_signal, peak_v_shoulder) = shoulder_getter(vg_df["V"],
                                                     vg_df["smoothed"])

    vcenter = peak_v_shoulder
    vstart = vcenter - 0.5*vwidth
    vend = vcenter + 0.5*vwidth

    detilter = make_detilter(vstart, vend, stiffness)
    vg_df["detilted"] = detilter(vg_df["V"].to_numpy(),
                                 vg_df["smoothed"].to_numpy())

    signal_getter = make_signal_getter(vstart, vend)
    (peak_signal_return, peak_v_return) = signal_getter(vg_df["V"], vg_df["detilted"])
    ymaxidx = numpy.argmax(vg_df["detilted"])

    peakarea = metrics.auc(vg_df["V"], vg_df["detilted"])*1000
    # print("peakarea", peakarea)
    return peakarea, peak_v_return, vg_df, vcenter, vg_df["detilted"][ymaxidx]

def find_first_derivative_peaks(V, Signal):
    V_center        = V[np.argmax(Signal)]
    
    spline_model    = scipy.interpolate.UnivariateSpline(V, Signal, s=0, k=4)
    spline_model_d  = spline_model.derivative(n=1)

    dS_dV           = spline_model_d(V) 
     
    dS_dV_max_peak  = dS_dV.max()                     # Maximum Peak Current
    dS_dV_min_peak  = dS_dV.min()                     # Minimum Peak Current
 
    dS_dV_max_V     = V[np.argmax(dS_dV)]             # Maximum Peak Volatage
    dS_dV_min_V     = V[np.argmin(dS_dV)]             # Minium Peak Voltage
    
    dS_dV_peak_diff = dS_dV_max_peak - dS_dV_min_peak # Finding the Peak difference

    dS_dV_spline    = scipy.interpolate.UnivariateSpline(V, dS_dV, s=0, k=4)
    dS_dV_area      = np.abs(dS_dV_spline.integral(V[0], V_center)) + np.abs(dS_dV_spline.integral(V_center, V[-1])) # Finding the Area of the First Derivative

    return V, dS_dV, dS_dV_max_peak, dS_dV_min_peak, dS_dV_peak_diff, dS_dV_max_V, dS_dV_min_V, dS_dV_area


def v2signal_extra_features(vg_filename: str,
             do_log: bool,
             smoothing_bw: float,
             vcenter: float,
             vwidth: float,
             stiffness: float):

    vg_df = read_raw_vg_as_df(vg_filename)

    if (vg_df['I'].to_numpy() < 0).any():
        return None, None, vg_df, None, None

    if do_log:
        cur_var_name = "logI"
        #vg_df[cur_var_name] = numpy.emath.logn(logbase, vg_df["I"])
        vg_df[cur_var_name] = numpy.log2(vg_df["I"])
    else:
        cur_var_name = "I"

    smoother = make_smoother(smoothing_bw)

    vg_df["smoothed"] = smoother(vg_df["V"], vg_df[cur_var_name].to_numpy())

    shoulder_getter = make_shoulder_getter(1, 1.1)
    (peak_signal, peak_v_shoulder) = shoulder_getter(vg_df["V"],
                                                     vg_df["smoothed"])

    vcenter = peak_v_shoulder
    vstart = vcenter - 0.5*vwidth
    vend = vcenter + 0.5*vwidth

    detilter = make_detilter(vstart, vend, stiffness)
    vg_df["detilted"] = detilter(vg_df["V"].to_numpy(),
                                 vg_df["smoothed"].to_numpy())

    signal_getter = make_signal_getter(vstart, vend)
    (peak_signal_return, peak_v_return) = signal_getter(vg_df["V"], vg_df["detilted"])
    ymaxidx = numpy.argmax(vg_df["detilted"])

    peakarea = metrics.auc(vg_df["V"], vg_df["detilted"])*1000

    V, dS_dV, dS_dV_max_peak, dS_dV_min_peak, dS_dV_peak_diff, \
    dS_dV_max_V, dS_dV_min_V, dS_dV_area        = find_first_derivative_peaks(vg_df["V"].values, vg_df["detilted"].values)
    
   
    return  peakarea, peak_signal_return, peak_v_return, vg_df, vcenter, vg_df["detilted"][ymaxidx],\
           dS_dV_max_peak, dS_dV_min_peak, dS_dV_peak_diff, dS_dV_max_V, dS_dV_min_V, dS_dV_area


if __name__ == '__main__':
    args = get_args()
    assert not (args.recenter and args.plot), \
        "Cannot specify both recenter and plot at the same time"

    vg_filename = args.filename
    vcenter = args.vcenter
    vwidth = args.vwidth
    assert vwidth > 0.0, f"vwidth must be nonnegative: {vwidth}"

    do_log = args.log

    smoothing_bw = args.bw
    assert smoothing_bw >= 0.0, "smoothing bandwidth must be " + \
        f"nonnegative: {smoothing_bw}"

    stiffness = args.smooth
    assert stiffness >= 0.0, "stiffness must be " + \
        f"nonnegative: {stiffness}"

    (peak_signal, peak_v, vg_df) = v2signal(vg_filename,
                                            do_log,
                                            smoothing_bw,
                                            vcenter,
                                            vwidth,
                                            stiffness)

    if peak_signal is not None:
        print(f"Peak voltage: {peak_v:0.3f} V")
        print(f"Signal: {peak_signal:0.3f} 1/V^2")
        if args.recenter:
            (peak_signal, peak_v, vg_df) = v2signal(vg_filename,
                                                    do_log,
                                                    smoothing_bw,
                                                    peak_v,
                                                    vwidth,
                                                    stiffness)
            if peak_signal is not None:
                print(f"Recentered peak voltage: {peak_v:0.3f} V")
                print(f"Recentered peak Signal: {peak_signal:0.3f} 1/V^2")
            else:
                print("no peak detected with recentered window; try --plot")
    else:
        print("no peak detected in original window; try running with --plot")

    if args.plot:
        plt.plot(vg_df["V"], vg_df["detilted"], "b-")
        plt.xlabel('baseline potential (V)')
        plt.ylabel('log peak current, normalized')
        plt.show()
