from pmtopt.geometry import calc_fom_confusion

TP_test = 66
FP_test = 0
FN_test = 44
TN_test = 89980
all = TP_test + FP_test + FN_test + TN_test
result = calc_fom_confusion(TP_test, FP_test, FN_test, all)
print("Figure of Merit Test: ", result)