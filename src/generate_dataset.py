import scipy
import os
import numpy as np
import pandas as pd
import sklearn
import scipy.stats as stats

from glob import glob

from src.vg2signal import read_raw_vg_as_df, make_smoother, make_detilter, make_signal_getter, make_shoulder_getter
from src.config import DATASET_PATH

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
        temp = [None] * 11
        return [None, None, vg_df] + temp

    if do_log:
        cur_var_name = "logI"
        #vg_df[cur_var_name] = np.emath.logn(logbase, vg_df["I"])
        vg_df[cur_var_name] = np.log2(vg_df["I"])
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

    # print(vg_df["detilted"].shape, vg_df["smoothed"].shape)

    signal_getter = make_signal_getter(vstart, vend)
    (peak_signal_return, peak_v_return) = signal_getter(vg_df["V"], vg_df["detilted"])
    ymaxidx = np.argmax(vg_df["detilted"])

    peakarea = sklearn.metrics.auc(vg_df["V"], vg_df["detilted"])*1000

    V, dS_dV, dS_dV_max_peak, dS_dV_min_peak, dS_dV_peak_diff, \
    dS_dV_max_V, dS_dV_min_V, dS_dV_area        = find_first_derivative_peaks(vg_df["V"].values, vg_df["detilted"].values)

    signal_std  = vg_df["detilted"].values.std()
    signal_mean = vg_df["detilted"].values.mean()
    
   
    return  peakarea, peak_signal_return, peak_v_return, vg_df, vcenter, vg_df["detilted"][ymaxidx], signal_mean, signal_std,\
           dS_dV_max_peak, dS_dV_min_peak, dS_dV_peak_diff, dS_dV_max_V, dS_dV_min_V, dS_dV_area

def make_xlsx_str(do_log, recenter, smoothing_bw, stiffness, vcenter, vwidth1, vwidth2):
    if do_log:  # if run with log-transform
        log_str = "_log"
    else:  # if not use log-transform
        log_str = "_NOlog"
    if recenter:  # if run with double detilt/ recentering
        recenter_str = "_recenter"
    else:  # if not recentering
        recenter_str = "_NOrecenter"

    smooth_str = "_" + str(smoothing_bw)
    stiff_str = "_" + str(stiffness)
    vcenter_str = "_" + str(vcenter)
    vwidth1_str = "_" + str(vwidth1)
    vwidth2_str = "_" + str(vwidth2)
    # combine all params into one string
    data_str = log_str + recenter_str + smooth_str + stiff_str + vcenter_str + vwidth1_str + vwidth2_str + "extra_features" ".xlsx"
    return data_str
    
def run_vg2(folderpath, do_log, recenter, smoothing_bw, stiffness, vcenter, vwidth1, vwidth2):
    # get filenames to save
    data_str = make_xlsx_str(do_log, recenter, smoothing_bw, stiffness, vcenter, vwidth1, vwidth2)
    vg_dict = dict()
    dfxl = pd.DataFrame()
    os.chdir(folderpath)  # change to desired folderpath
    signal_lst = []
    conc_dict = dict()  # [cbz concentration]: peak signals
    for filename in os.listdir():
        if filename[-3:] == 'txt':
            print("Analyzing:", filename)
            (peak_signal, peak_curvature, peak_v, vg_df, vcenter, ph, signal_mean, signal_std,\
dS_dV_max_peak, dS_dV_min_peak, dS_dV_peak_diff, dS_dV_max_V, dS_dV_min_V, dS_dV_area) = v2signal_extra_features(filename,
                                                                                   do_log,
                                                                                   smoothing_bw,
                                                                                   vcenter,
                                                                                   vwidth1,
                                                                                   stiffness)
            
            if (peak_signal == None) or (peak_curvature==None):
                print(f"peak_signal:{peak_signal} OR peak curvature: {peak_curvature}", filename)
                continue

            idx1 = filename.rfind("cbz")
            idx2 = filename[idx1:].find("_")
            conc = filename[idx1 + 3:idx1 + idx2]
            replicate = filename[idx1 + idx2 + 1:filename.rfind(".")]

            if 'p' in conc:  # for 7p5 concentration
                pi = conc.find('p')
                conctemp = conc[:pi] + '.' + conc[pi + 1:]
                conc = conctemp
            concstrxl = str(float(conc))
            concxl = list([concstrxl] * len(vg_df["V"]))
            replicatexl = list([replicate] * len(vg_df["V"]))
            if do_log:
                dfxl = pd.concat([dfxl, pd.DataFrame(
                    [concxl, replicatexl, vg_df["V"], vg_df["I"], vg_df["logI"], vg_df["smoothed"],
                     vg_df["detilted"]]).transpose()])
            else:
                dfxl = pd.concat([dfxl, pd.DataFrame(
                    [concxl, replicatexl, vg_df["V"], vg_df["I"], vg_df["smoothed"], vg_df["detilted"]]).transpose()])

            if peak_signal is None:  # if find no peak
                peak_signal = 0
            if peak_v is None:
                peak_v = 0
            signal_lst.append([filename, round(peak_signal, 4), round(peak_curvature, 4), round(peak_v, 4), round(vcenter, 4), ph, round(signal_mean, 4), round(signal_std, 4),\
                              round(dS_dV_max_peak, 4), round(dS_dV_min_peak, 4), round(dS_dV_peak_diff, 4), round(dS_dV_max_V, 4), round(dS_dV_min_V, 4), round(dS_dV_area, 4)])  # add text filename & peak signal to signal list
            if conc in conc_dict.keys():  # for each concentration
                conclst = conc_dict[conc]
                conclst.append((peak_signal, peak_v))  # add peak signal to concentration dictionary
                conc_dict[conc] = conclst

                # for plotting purposes
                plst = vg_dict[conc]
                plst.append(vg_df)
                vg_dict[conc] = plst
            else:
                conc_dict[conc] = [(peak_signal, peak_v)]
                vg_dict[conc] = [vg_df]

    signal_df = pd.DataFrame(signal_lst)
    conc_list = []
    concs_targetlst = sorted([c for idx, c in enumerate(list(conc_dict.keys()))], key=lambda v: float(v))

    for key in conc_dict:  # for each concentration
        vals = conc_dict[key]  # all the signals for conc
        avgval = round(np.average([val[0] for val in vals]), 2)  # avg signal for conc
        stdval = round(np.std([val[0] for val in vals]), 2)  # std of signals for conc
        avgpeakval = round(np.average([val[1] for val in vals]), 2)  # avg peak voltage for conc
        stdpeakval = round(np.std([val[1] for val in vals]), 2)  # std of peak voltage for conc
        if avgval != 0:
            cvval = round(stdval / avgval, 3)
        else:
            cvval = 0  # if average is 0, make CV 0
        concstr = str(float(key)) + " \u03BCM"
        # compare signal list for this conc to closest lower conc
        currentidx = concs_targetlst.index(key)
        if currentidx == 0:
            lowervals = conc_dict[key]
        else:
            lowervals = conc_dict[concs_targetlst[currentidx-1]]
        ttest = round(stats.ttest_ind([val[0] for val in vals], [val[0] for val in lowervals], equal_var=False)[0], 2)
        conc_list.append([concstr, avgval, stdval, cvval, ttest, avgpeakval, stdpeakval])  # add stats for conc

    conc_lst_sorted = sorted(conc_list, key=lambda x: float(x[0][:-2]))
    conc_df = pd.DataFrame(conc_lst_sorted)
    
    # save stats list to excel
    stats_str = "stats" + data_str
    signal_str = "extracted_features.xlsx"
    dataframe_str = "dataframe" + data_str
    conc_df.to_excel(stats_str, index=False,
                     header=["conc", "average", "std", "CV", "T-Statistic", "avg peak", "std peak"])
    signal_df.to_excel(signal_str, index=False,
                       header=["file", "peak area", "peak curvature", "peak V", "vcenter", "PH", "signal_mean", "signal_std", \
                              "dS_dV_max_peak", "dS_dV_min_peak", "dS_dV_peak_diff", "dS_dV_max_V", "dS_dV_min_V", "dS_dV_area"])  # save signal list to excel
    if do_log:
        dfxl.to_excel(dataframe_str, index=False,
                      header=["conc", "replicate", "V", "I", "logI", "smoothed", "detilted"])
    else:
        dfxl.to_excel(dataframe_str, index=False, header=["conc", "replicate", "V", "I", "smoothed", "detilted"])
    
    return vg_dict, data_str

if __name__ == '__main__':
    all_dataset = glob(f"{DATASET_PATH}/*")
    
    do_log   = True
    recenter = False
    bw       = 0.006
    s        = 0
    c        = 1.04
    w1       = 0.15
    w2       = 0.17
    
    for dataset_path in all_dataset:
        conc_df, signal_df = run_vg2(dataset_path, do_log, recenter, bw, s, c, w1, w2)