import matplotlib.pyplot as plt
import os
import csv
import numpy as np

cond = 'cnp'
cond_expid = 'rbs00'
lat = 'np'
lat_expid = 'rbs00'
boot = 'bnp'
boot_expid = 'trial'

root = '/mnt/aitrics_ext/ext01/john/neural_process'
r_bs_vals = np.arange(0.0, 0.9, 0.1)

cond_lls = []
lat_lls = []
boot_lls = []
for r_bs in r_bs_vals:
    with open(os.path.join(root, cond, 'rbf', cond_expid,
        'rbf/{:.1f}_eval.log'.format(r_bs)), 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        line = next(iter(reader))
        cond_lls.append(float(line[-3]))

    with open(os.path.join(root, lat, 'rbf', lat_expid,
        'rbf/{:.1f}_eval.log'.format(r_bs)), 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        line = next(iter(reader))
        lat_lls.append(float(line[-3]))

    with open(os.path.join(root, boot, 'rbf', boot_expid,
        'rbf/{:.1f}_eval.log'.format(r_bs)), 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        line = next(iter(reader))
        boot_lls.append(float(line[-3]))

plt.figure('RBF')
plt.plot(r_bs_vals, cond_lls, label=cond.upper(), marker='o',
        markeredgecolor='royalblue', markerfacecolor='ghostwhite',
        color='lavender', markersize=10, linewidth=2)
plt.plot(r_bs_vals, lat_lls, label=lat.upper(), marker='d',
        markeredgecolor='indianred', markerfacecolor='mistyrose',
        color='lightcoral', markersize=10, linewidth=2)
plt.plot(r_bs_vals, boot_lls, label=boot.upper(), marker='h',
        markeredgecolor='mediumseagreen', markerfacecolor='mintcream',
        color='springgreen', markersize=10, linewidth=2)
plt.legend()

cond_lls = []
lat_lls = []
boot_lls = []
for r_bs in r_bs_vals:
    with open(os.path.join(root, cond, 'rbf', cond_expid,
        'periodic/{:.1f}_eval.log'.format(r_bs)), 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        line = next(iter(reader))
        cond_lls.append(float(line[-3]))

    with open(os.path.join(root, lat, 'rbf', lat_expid,
        'periodic/{:.1f}_eval.log'.format(r_bs)), 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        line = next(iter(reader))
        lat_lls.append(float(line[-3]))

    with open(os.path.join(root, boot, 'rbf', boot_expid,
        'periodic/{:.1f}_eval.log'.format(r_bs)), 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        line = next(iter(reader))
        boot_lls.append(float(line[-3]))

plt.figure('Periodic')
plt.plot(r_bs_vals, cond_lls, label=cond.upper(), marker='o',
        markeredgecolor='royalblue', markerfacecolor='ghostwhite',
        color='lavender', markersize=10, linewidth=2)
plt.plot(r_bs_vals, lat_lls, label=lat.upper(), marker='d',
        markeredgecolor='indianred', markerfacecolor='mistyrose',
        color='lightcoral', markersize=10, linewidth=2)
plt.plot(r_bs_vals, boot_lls, label=boot.upper(), marker='h',
        markeredgecolor='mediumseagreen', markerfacecolor='mintcream',
        color='springgreen', markersize=10, linewidth=2)
plt.legend()

cond_lls = []
lat_lls = []
boot_lls = []
for r_bs in r_bs_vals:
    with open(os.path.join(root, cond, 'rbf', cond_expid,
        'htn/{:.1f}_eval.log'.format(r_bs)), 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        line = next(iter(reader))
        cond_lls.append(float(line[-3]))

    with open(os.path.join(root, lat, 'rbf', lat_expid,
        'htn/{:.1f}_eval.log'.format(r_bs)), 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        line = next(iter(reader))
        lat_lls.append(float(line[-3]))

    with open(os.path.join(root, boot, 'rbf', boot_expid,
        'htn/{:.1f}_eval.log'.format(r_bs)), 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        line = next(iter(reader))
        boot_lls.append(float(line[-3]))

plt.figure('RBF + Heavy tailed noise')
plt.plot(r_bs_vals, cond_lls, label=cond.upper(), marker='o',
        markeredgecolor='royalblue', markerfacecolor='ghostwhite',
        color='lavender', markersize=10, linewidth=2)
plt.plot(r_bs_vals, lat_lls, label=lat.upper(), marker='d',
        markeredgecolor='indianred', markerfacecolor='mistyrose',
        color='lightcoral', markersize=10, linewidth=2)
plt.plot(r_bs_vals, boot_lls, label=boot.upper(), marker='h',
        markeredgecolor='mediumseagreen', markerfacecolor='mintcream',
        color='springgreen', markersize=10, linewidth=2)
plt.legend()
plt.show()

## rbf
#with open(os.path.join(root, cond, 'rbf', 'trial', 'rbf_eval.log'), 'r') as f:
#    reader = csv.reader(f, delimiter=' ')
#    line = next(iter(reader))
#    cond_ll = float(line[-3][:-1])
#
#with open(os.path.join(root, lat, 'rbf', 'trial', 'rbf_eval.log'), 'r') as f:
#    reader = csv.reader(f, delimiter=' ')
#    line = next(iter(reader))
#    lat_ll = float(line[-3][:-1])
#
#boot_lls = []
#for r in rs:
#    with open(os.path.join(root, boot, 'rbf', 'trial',
#        'rbf_eval_{}.log'.format(r)), 'r') as f:
#        reader = csv.reader(f, delimiter=' ')
#        line = next(iter(reader))
#        boot_lls.append(float(line[-3][:-1]))
#boot_lls = np.array(boot_lls)
#
#plt.figure('RBF')
#plt.plot(1-rs, boot_lls, 'o-', label=boot.upper(),
#        markeredgecolor='royalblue', markerfacecolor=None,
#        color='lavender', markersize=10, linewidth=2)
#plt.axhline(cond_ll, linestyle='--', color='lightcoral', label=cond.upper(), linewidth=2)
#plt.axhline(lat_ll, linestyle='-.', color='olive', label=lat.upper(), linewidth=2)
#plt.legend(fontsize=20)
#plt.xticks(fontsize=15)
#plt.xlabel(r'$r_{ \mathrm{bs} }$', fontsize=20)
#plt.yticks(fontsize=15)
#plt.ylabel('Pred LL', fontsize=20)
#maxval = max(boot_lls.max(), cond_ll, lat_ll)
#minval = min(boot_lls.min(), cond_ll, lat_ll)
#plt.ylim([minval-abs(minval)*0.1, maxval+abs(maxval)*0.1])
#plt.savefig('rbf_{}.pdf'.format(boot), bbox_inches='tight')
#
## periodic
#with open(os.path.join(root, cond, 'rbf', 'trial', 'periodic_eval.log'), 'r') as f:
#    reader = csv.reader(f, delimiter=' ')
#    line = next(iter(reader))
#    cond_ll = float(line[-3][:-1])
#
#with open(os.path.join(root, lat, 'rbf', 'trial', 'periodic_eval.log'), 'r') as f:
#    reader = csv.reader(f, delimiter=' ')
#    line = next(iter(reader))
#    lat_ll = float(line[-3][:-1])
#
#boot_lls = []
#for r in rs:
#    with open(os.path.join(root, boot, 'rbf', 'trial',
#        'periodic_eval_{}.log'.format(r)), 'r') as f:
#        reader = csv.reader(f, delimiter=' ')
#        line = next(iter(reader))
#        boot_lls.append(float(line[-3][:-1]))
#boot_lls = np.array(boot_lls)
#
#plt.figure('Periodic')
#plt.plot(1-rs, boot_lls, 'o-', label=boot.upper(),
#        markeredgecolor='royalblue', markerfacecolor=None,
#        color='lavender', markersize=10, linewidth=2)
#plt.axhline(cond_ll, linestyle='--', color='lightcoral', label=cond.upper(), linewidth=2)
#plt.axhline(lat_ll, linestyle='-.', color='olive', label=lat.upper(), linewidth=2)
#plt.legend(fontsize=20)
#plt.xticks(fontsize=15)
#plt.xlabel(r'$r_{ \mathrm{bs} }$', fontsize=20)
#plt.yticks(fontsize=15)
#plt.ylabel('Pred LL', fontsize=20)
#maxval = max(boot_lls.max(), cond_ll, lat_ll)
#minval = min(boot_lls.min(), cond_ll, lat_ll)
#plt.ylim([minval-abs(minval)*0.1, maxval+abs(maxval)*0.1])
#plt.savefig('periodic_{}.pdf'.format(boot), bbox_inches='tight')
#
## heavy-tailed noise
#with open(os.path.join(root, cond, 'rbf', 'trial', 'rbf_htn_eval.log'), 'r') as f:
#    reader = csv.reader(f, delimiter=' ')
#    line = next(iter(reader))
#    cond_ll = float(line[-3][:-1])
#
#with open(os.path.join(root, lat, 'rbf', 'trial', 'rbf_htn_eval.log'), 'r') as f:
#    reader = csv.reader(f, delimiter=' ')
#    line = next(iter(reader))
#    lat_ll = float(line[-3][:-1])
#
#boot_lls = []
#for r in rs:
#    with open(os.path.join(root, boot, 'rbf', 'trial',
#        'rbf_htn_eval_{}.log'.format(r)), 'r') as f:
#        reader = csv.reader(f, delimiter=' ')
#        line = next(iter(reader))
#        boot_lls.append(float(line[-3][:-1]))
#boot_lls = np.array(boot_lls)
#
#plt.figure('RBF + Heavy tailed noise')
#plt.plot(1-rs, boot_lls, 'o-', label=boot.upper(),
#        markeredgecolor='royalblue', markerfacecolor=None,
#        color='lavender', markersize=10, linewidth=2)
#plt.axhline(cond_ll, linestyle='--', color='lightcoral', label=cond.upper(), linewidth=2)
#plt.axhline(lat_ll, linestyle='-.', color='olive', label=lat.upper(), linewidth=2)
#plt.legend(fontsize=20)
#plt.xticks(fontsize=15)
#plt.xlabel(r'$r_{ \mathrm{bs} }$', fontsize=20)
#plt.yticks(fontsize=15)
#plt.ylabel('Pred LL', fontsize=20)
#maxval = max(boot_lls.max(), cond_ll, lat_ll)
#minval = min(boot_lls.min(), cond_ll, lat_ll)
#plt.ylim([minval-abs(minval)*0.1, maxval+abs(maxval)*0.1])
#plt.savefig('rbf_htn_{}.pdf'.format(boot), bbox_inches='tight')
#
#plt.show()
