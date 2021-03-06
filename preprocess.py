import heartpy
import imageio
import matplotlib.pyplot as plt
import numpy as np
from heartpy.datautils import rolling_mean
from sklearn import preprocessing
from filters import bandpass_butter
import cv2
from numpy.linalg import inv, norm, lstsq
from numpy.linalg import matrix_rank as rank


# from model.apply_model import apply_model_df


def crop_center(img, ww=640, hh=480):
    h, w, _ = img.shape
    w_st = w // 2 - (ww // 2)
    h_st = h // 2 - (hh // 2)
    return img[h_st:h_st + ww, w_st:w_st + hh]


def img_to_signal(img, crop=False):
    if crop:
        img = crop_center(img)

    # fn = np.sum
    fn = np.mean

    r, g, b = img[:, :, 0], img[:, :, 0], img[:, :, 0]
    out = [fn(r), fn(g), fn(b)]
    return out


def process_data(x, fps=20):
    #print(x.shape)
    #x = x[5:-5]

    x = preprocessing.scale(x)
    x = bandpass_butter(x, cut_low=1, cut_high=2, rate=fps, order=2)

    rol_mean = rolling_mean(x, windowsize=0.75, sample_rate=fps)
    wd = heartpy.peakdetection.detect_peaks(x, rol_mean, ma_perc=20, sample_rate=fps)
    peaks = wd['peaklist']
    rri = np.diff(peaks)
    rri = rri * 1 / fps * 1000
    hr = 6e4 / rri

    hr = heartpy.datautils.outliers_iqr_method(hr)
    hr = np.asarray(hr[0])
    hr = heartpy.filtering.smooth_signal(hr, sample_rate=fps, window_length=4, polyorder=2)

    hr[hr > 100] = 100
    hr[hr < 40] = 40

    # df = pd.DataFrame()
    # df['rri'] = [rri]
    # res = apply_model_df(df)
    # print(f"res: {res}")

    # m, wd = heartpy.analysis.calc_breathing(wd['RR_list'], x, sample_rate=fps)
    # print(f"breath rate: {round(m['breathingrate'], 3)}")

    '''fig, axs = plt.subplots(3, 1, figsize=(17, 9))
    axs = axs.ravel()
    axs[0].plot(x, '-b', label='x')
    axs[0].plot(peaks, x[peaks], 'r.', label='peaks')
    axs[1].plot(rri, '-r.', label='rri')
    axs[2].plot(hr, '-r.', label='hr')
    for ax in axs:
        ax.legend()
    plt.show()'''
    return hr

def process_data_ref(x, fps=30):
    #print(x.shape)
    x = x[5:-5]

    x = preprocessing.scale(x)
    x = bandpass_butter(x, cut_low=1, cut_high=2, rate=fps, order=2)

    rol_mean = rolling_mean(x, windowsize=3, sample_rate=fps)
    wd, m = heartpy.process_segmentwise(rol_mean, sample_rate = fps, segment_width=3, segment_overlap=0.5,
                                        replace_outliers = True)
    bpm = [round(x, 1) for x in m['bpm']]
    print(bpm)

    '''from scipy.interpolate import make_interp_spline
    x_t = np.asarray(range(0, len(hr)))
    x_smooth = np.linspace(x_t.min(), x_t.max(), 300)
    hr = make_interp_spline(x_t, hr)(x_smooth)'''

    # df = pd.DataFrame()
    # df['rri'] = [rri]
    # res = apply_model_df(df)
    # print(f"res: {res}")

    # m, wd = heartpy.analysis.calc_breathing(wd['RR_list'], x, sample_rate=fps)
    # print(f"breath rate: {round(m['breathingrate'], 3)}")

    return bpm

def show_frame(vid):
    for idx, frame in enumerate(vid.iter_data()):
        idxx = idx % 30
        if idxx == 0:
            fig, axs = plt.subplots(5, 6, figsize=(17, 9))
            axs = axs.ravel()

        axs[idxx].imshow(frame)
        axs[idxx].set_title(f"frame id: {idx}")
        axs[idxx].axis('off')
        if idxx % 29 == 0 and idxx != 0:
            plt.tight_layout()
            plt.show()


def process_frame(data=None, fps=30):
    fp = data
    #vid = imageio.get_reader(fp, 'ffmpeg')
    #metadata = vid.get_meta_data()
    #fps = metadata['fps']
    rgb = [img_to_signal(img) for img in fp]
    rgb = np.vstack(rgb)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    hr = process_data_ref(r, fps)
    return hr


def align_faces(uv, xy=None, K=2):

    REFERENCE_FACIAL_POINTS = np.array([
        [30.29459953, 51.69630051],
        [65.53179932, 51.50139999],
        [48.02519989, 71.73660278],
        [33.54930115, 92.3655014],
        [62.72990036, 92.20410156]
    ], np.float32)

    uv = np.squeeze(uv)
    xy = REFERENCE_FACIAL_POINTS
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))

    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    U = np.vstack((u, v))

    # We know that X * r = U
    if rank(X) >= 2 * K:
        r, _, _, _ = lstsq(X, U)
        r = np.squeeze(r)
    else:
        raise Exception('cp2tform:twoUniquePointsReq')

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    Tinv = np.array([
        [sc, -ss, 0],
        [ss,  sc, 0],
        [tx,  ty, 1]
    ])


    T = inv(Tinv)

    T[:, 2] = np.array([0, 0, 1])

    T = T[:, 0:2].T

    return T

def aligned_face(landmark, fr, fig_size):
    REFERENCE_FACIAL_POINTS = np.array([
        [30.29459953, 51.69630051],
        [65.53179932, 51.50139999],
        [48.02519989, 71.73660278],
        [33.54930115, 92.3655014],
        [62.72990036, 92.20410156]
    ], np.float32)
    trans_matrix = cv2.getAffineTransform(landmark[0][:3], REFERENCE_FACIAL_POINTS[:3])
    aligned_face = cv2.warpAffine(fr.copy(), trans_matrix, fig_size)
    return aligned_face
def eliminate_light(fr):
    lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    grayimg1 = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)
    mask2 = cv2.threshold(grayimg1, 220, 255, cv2.THRESH_BINARY)[1]
    result2 = cv2.inpaint(fr, mask2, 0.1, cv2.INPAINT_TELEA)
    return result2




if __name__ == "__main__":
    fp = r'.\Savevideo\result_right.avi'
    vid = imageio.get_reader(fp, 'ffmpeg')
    metadata = vid.get_meta_data()
    # print(f'metadata: {metadata}')
    fps = metadata['fps']
    # show_frame(vid)

    rgb = [img_to_signal(img) for img in vid.iter_data()]
    rgb = np.vstack(rgb)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    process_data_ref(r, fps)
