
import argparse
import torch, cv2, celldetection as cd
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
import pandas as pd
import os

def draw_contours_(canvas, contours, close=True):
    """Draw contours.

    Draw ``contours`` on ``canvas``.

    Note:
        This is an inplace operation.

    Args:
        canvas: Tensor[h, w].
        contours: Contours in (x, y) format. Tensor[num_contours, num_points, 2].
        close: Whether to close contours. This is necessary if the last point of a contour is not equal to the first.

    """
    if close:
        contours = torch.cat((contours, contours[..., :1, :]), -2)
    contours = contours.to(torch.int32)
    diff = torch.diff(contours, axis=1)
    sign, diff = torch.sign(diff), torch.abs(diff)
    err = diff[..., 0] - diff[..., 1]
    x, y = contours[:, :-1, 0] + 0, contours[:, :-1, 1] + 0  # start point
    x_, y_ = contours[:, 1:, 0], contours[:, 1:, 1]  # end point
    labels = torch.broadcast_to(torch.arange(1, 1 + len(contours), device=canvas.device).to(canvas.dtype)[:, None],
                                x.shape)
    m = torch.ones(x.shape, dtype=torch.bool, device=canvas.device)
    while True:
        canvas[y[m], x[m]] = labels[m]
        # m: Select lines that are not finished; m_: Remove contours that are finished
        m = m & (((x != x_) | (y != y_)) & (x >= 0) & (y >= 0) & (x < canvas.shape[1]) & (y < canvas.shape[0]))
        m_ = torch.any(m, axis=-1)
        m = m[m_]
        if len(m) <= 0:
            break
        x, y, x_, y_, err, diff, sign, labels = (i[m_] for i in (x, y, x_, y_, err, diff, sign, labels))
        err_ = 2 * err
        sel = err_ > -diff[..., 1]
        err[sel] -= diff[sel][..., 1]  # Note: torch cannot handle arr[mask, index] properly; not equivalent to numpy
        x[sel] += sign[sel][..., 0]
        sel = err_ < diff[..., 0]
        err[sel] += diff[sel][..., 0]
        y[sel] += sign[sel][..., 1]

    return canvas

def analyze(fig_path):
    print("Processing %s ... " % fig_path)
    fig_name = os.path.basename(fig_path)
    
    ## download and load pretrained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Running on %s ... " % device)
    print("Loading model ... ")
    model = cd.fetch_model("ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c", check_hash=True)
    model = model.to(device)
    model.eval()

    ## Load input image
    img = cv2.imread(fig_path)
    ## crop the image to h
    h, w = img.shape[:2]
    if h > w:
        img = img[:w, :, :]
    else:
        img = img[:, :h, :]
    ## BGR2RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run model
    with torch.no_grad():
        print("Running model ... ")
        xx = cd.to_tensor(img, transpose=True, device=device, dtype=torch.float32)
        xx = xx / 255  # ensure 0..1 range
        xx = xx[None]  # add batch dimension: Tensor[3, h, w] -> Tensor[1, 3, h, w]
        yy = model(xx)
    contours = yy['contours']
    contour_arrays = contours[0].detach().cpu()
    canvas = draw_contours_(torch.zeros(xx.detach().cpu().shape[2:]), contour_arrays).numpy()
    canvas = cv2.cvtColor(canvas / canvas.max() * 255, cv2.COLOR_GRAY2RGB)
    img[canvas > 0] = 100

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot()
    plt.imshow(img)
    
    ## Plot contour image and do statistics
    contours_size =  contour_arrays / img.shape[0] * args.units
    cell_areas = np.zeros(contour_arrays.shape[0])
    for i in range(contour_arrays.shape[0]):
        area = cv2.contourArea(contours_size[i].numpy())
        cell_areas[i] = area
        x_c, y_c = contour_arrays[i][:,0].mean(), contour_arrays[i][:,1].mean()
        ax.text(x_c, y_c, int(area), fontsize=15)
    # plt.savefig("logs/seg_" + args.fig)
    output_fig_path = os.path.join(args.output, "seg_" + fig_name)
    plt.savefig(output_fig_path)
    plt.close()

    ## calculate mean and standard deviation
    diamaters = np.sqrt(cell_areas / np.pi) * 2
    mu, std = norm.fit(diamaters)
    x_norm = np.linspace(0, diamaters.max(), 100)
    p = norm.pdf(x_norm, mu, std)
    
    ## Dump statistics information to csv
    df = pd.DataFrame({'area': cell_areas, 'diameter': diamaters})
    # df.to_csv("logs/areas_" + args.prefix + ".csv", index=False)
    output_csv_path = os.path.join(args.output, fig_name + ".csv")
    df.to_csv(output_csv_path, index=False)
    print("mu = ", mu, "std = ", std)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('cellseg')
    parser.add_argument('--fig', type=str, default='', help='')
    parser.add_argument('--folder', type=str, default='', help='')
    parser.add_argument('--output', type=str, default='logs', help='')
    parser.add_argument('--units', type=float, default=1000, help='nanometers')
    args = parser.parse_args()
    args.prefix = args.fig.split('.')[0]
    ## create output directory if not exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    ## args.fig and args.folder can only be specified once
    if args.fig != '':
        analyze(args.fig)
    elif args.folder != '':
        ## list all .png and .jpg images
        files = os.listdir(args.folder)
        files = [f for f in files if f.endswith('.png') or f.endswith('.jpg')]
        for f in files:
            fig = os.path.join(args.folder, f)
            analyze(fig)

