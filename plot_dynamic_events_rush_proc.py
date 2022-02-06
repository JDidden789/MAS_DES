import pandas as pd
from matplotlib import pyplot as plt, rcParams
import numpy as np
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

utilization = [85, 85, 90, 90, 95, 95]
due_date_settings = [4, 6, 4, 6, 4, 6]

fig, ax = plt.subplots(5, 2)
Names = ["0.00", "0.05", "0.10", "0.15", "0.20"]

for i in range(len(utilization)):
    filename = 'Results/Rush_Job' + str(utilization[i]) + '_' + str(due_date_settings[i]) + '.csv'
    filename1 = 'Results/Rush_Job' + str(utilization[i]) + '_' + str(due_date_settings[i]) + '_Static' + '.csv'

    filename2 = 'Results/Proc_Job' + str(utilization[i]) + '_' + str(due_date_settings[i]) + '.csv'
    filename3 = 'Results/Proc_Job' + str(utilization[i]) + '_' + str(due_date_settings[i]) + '_Static' + '.csv'

    data = pd.read_csv(filename, index_col=0)
    df = pd.DataFrame(data)

    data = pd.read_csv(filename1, index_col=0)
    df1 = pd.DataFrame(data)

    data = pd.read_csv(filename2, index_col=0)
    df2 = pd.DataFrame(data)

    data = pd.read_csv(filename3, index_col=0)
    df3 = pd.DataFrame(data)

    ax[0, 0].plot(df['Robustness'] / max(df1['Robustness']), 'o-')
    ax[0, 0].set_ylabel("Relative objective vs. RSS")
    ax[0, 0].grid(b=True, which='both')
    ax[0, 0].set_yticks(np.arange(0, 2, 0.25))
    ax[0, 0].set_yticks(np.arange(0, 2, 0.25), minor=True)

    ax[1, 0].plot(df['Stability 1'] / max(df1['Stability 1']), 'o-')
    ax[1, 0].set_ylabel("Relative increase in objective vs. RSS")
    ax[1, 0].grid(b=True, which='both')
    ax[1, 0].set_yticks(np.arange(0, 1.25, 0.25))
    ax[1, 0].set_yticks(np.arange(0, 1.25, 0.25), minor=True)

    ax[2, 0].plot(df['Stability 2'], 'o-')
    ax[2, 0].set_ylabel("Increase in processing time")
    ax[2, 0].grid(b=True, which='both')
    ax[2, 0].set_yticks(np.arange(0, 3000, 500))
    ax[2, 0].set_yticks(np.arange(0, 2750, 250), minor=True)

    ax[3, 0].plot(df['Stability 3'], 'o-')
    ax[3, 0].set_ylabel("Increase in setup time")
    ax[3, 0].grid(b=True, which='both')
    ax[3, 0].set_yticks(np.arange(0, 150, 25))
    ax[3, 0].set_yticks(np.arange(0, 150, 25), minor=True)

    ax[4, 0].plot(df['Stability 4'], 'o-')
    ax[4, 0].set_ylabel("Increase in job switches")
    ax[4, 0].grid(b=True, which='both')
    ax[4, 0].set_yticks(np.arange(0, 3100, 500))
    ax[4, 0].set_yticks(np.arange(0, 3100, 250), minor=True)
    ax[4, 0].set_xlabel(r"$\rho_r$", fontsize=14)

    ax[0, 1].plot(df2['Robustness'] / max(df3['Robustness']), 'o-')
    ax[0, 1].grid(b=True, which='both')
    ax[0, 1].set_yticks(np.arange(0, 5.5, 1))
    ax[0, 1].set_yticks(np.arange(0, 5.5, 0.5), minor=True)

    ax[1, 1].plot(df2['Stability 1'] / max(df3['Stability 1']), 'o-')
    ax[1, 1].grid(b=True, which='both')
    ax[1, 1].set_yticks(np.arange(0, 11, 2))
    ax[1, 1].set_yticks(np.arange(0, 11, 1), minor=True)

    ax[2, 1].plot(df2['Stability 2'], 'o-')
    ax[2, 1].grid(b=True, which='both')
    ax[2, 1].set_yticks(np.arange(0, 6100, 1000))
    ax[2, 1].set_yticks(np.arange(0, 6100, 500), minor=True)

    ax[3, 1].plot(df2['Stability 3'], 'o-')
    ax[3, 1].grid(b=True, which='both')
    ax[3, 1].set_yticks(np.arange(0, 270, 50))
    ax[3, 1].set_yticks(np.arange(0, 2, 25), minor=True)

    ax[4, 1].plot(df2['Stability 4'], 'o-')
    ax[4, 1].set_xlabel(r"$\rho_p/\kappa_p$", fontsize=14)
    ax[4, 1].grid(b=True, which='both')
    ax[4, 1].set_yticks(np.arange(0, 3100, 500))
    ax[4, 1].set_yticks(np.arange(0, 3100, 250), minor=True)

for i in range(5):
    plt.setp(ax[i, 0], xticks=range(4), xticklabels=['0.05', '0.10', '0.15', '0.20'])
    plt.setp(ax[i, 1], xticks=range(8), xticklabels=['0.05/0.05', '0.05/0.10', '0.05/0.15', '0.05/0.20', '0.10/0.05', '0.10/0.10', '0.10/0.15', '0.10/0.20'])


# plt.setp(ax, xticks=[0, 3, 4, 7, 8, 11], xticklabels=['0.05/1', '0.20/1', '0.05/5', '0.20/5', '0.05/10', '0.20/10'])
fig.suptitle('Rush Jobs and Jobs with Processing Time Increase', fontsize=24)
measure = [r"$\mathcal{R}^{MAS}/\mathcal{R}^{RSS}$", r"$\mathcal{S}^{MAS}_1/\mathcal{S}^{RSS}_1$", r"$\mathcal{S}^{MAS}_2$", r"$\mathcal{S}^{MAS}_3$", r"$\mathcal{S}^{MAS}_4$"]
due_date_tightness = ["Rush Jobs", "Increase in Processing Time"]
for i, row in enumerate(ax):
    for j, cell in enumerate(row):
        print(j)
        if i == 0:
            cell.set_title(due_date_tightness[j], fontsize=14)
        if j == 0:
            # print(i)
            cell.set_ylabel(measure[i], fontsize=14)

ax[0, 1].legend(["85-4", "85-6", "90-4", "90-6", "95-4", "95-6"], loc='upper right', bbox_to_anchor=(1.2, 1.0))
plt.show()
