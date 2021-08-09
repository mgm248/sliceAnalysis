import os

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

ffolder = r'C:\Users\Michael\Analysis\myRecordings_extra\21-05-10\filt_and_rsamp\\'
Fs = 20000
rfs = 1000
elecs = ['A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'B10', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'C10', 'C11', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'D1', 'D10', 'D11', 'D12', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'E1', 'E10', 'E11', 'E12', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'F1', 'F10', 'F11', 'F12', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'G1', 'G10', 'G11', 'G12', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'H1', 'H10', 'H11', 'H12', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'J1', 'J10', 'J11', 'J12', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'K10', 'K11', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8', 'K9', 'L10', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']
for fname in os.listdir(ffolder):
    if '.npy' in fname:
        data_dict = np.load(ffolder+ fname,allow_pickle=True)
        data = np.empty((len(elecs), data_dict.item()['A4'].shape[0]))
        i = 0
        for elec in elecs:
            data[i, :] = data_dict.item()[elec]
            i += 1


        pca = PCA(n_components=1)
        H = pca.fit_transform(data.T)
        X_pca = pca.transform(data.T)

        data = data - X_pca.T
        for elec in elecs:
            data_dict.item()[elec] = np.squeeze(data_dict.item()[elec] - X_pca.T)
        print('done')

        np.save(ffolder+'PCA_removed\\'+fname,data_dict)

# times=[6000,9000]
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(data[12,times[0]:times[1]])
# plt.title('Example electrode')
# plt.subplot(2,1,2)
# plt.plot(X_pca[times[0]:times[1]])
# plt.title('PCA')