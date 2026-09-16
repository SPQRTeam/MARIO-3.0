from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from adjustText import adjust_text

# that's way too many points otherwise!
SUBSAMPLE_RATE = 200
DOT_SIZE = 25
Y_MAX = 3200
ANNOTATE_OUTLIERS = False

FPS = 30
SNAP = 5 * FPS

# currently as a window average
def produce_subsampled_xy(data, subsample_rate):
    data = np.ma.masked_invalid(data)
    num_windows = len(data) // subsample_rate
    remainder_count = len(data) % subsample_rate

    # process regular-size windows
    data_main = data[:num_windows * subsample_rate]
    averaged_data = np.mean(data_main.reshape(-1, subsample_rate), axis=1)
    std_data = np.std(data_main.reshape(-1, SUBSAMPLE_RATE), axis=1)
    x_frames = np.arange(0, len(data_main), subsample_rate)

    # process the leftover tail
    if remainder_count > 0:
        tail_start_index = num_windows * subsample_rate
        tail_data = data[tail_start_index:]
        averaged_data = np.append(averaged_data, np.mean(tail_data))
        std_data = np.append(std_data, np.std(tail_data))
        x_frames = np.append(x_frames, tail_start_index)

    return x_frames, averaged_data, std_data

def plot(ax, x, y, y_err, y_max, annotate_outliers):
    inlier_mask = y <= y_max
    outlier_mask = y > y_max

    # plot the inliers and their error bars normally
    plt.scatter(x[inlier_mask], y[inlier_mask], color="blue", edgecolors="k", s=15, zorder=3)
    plt.errorbar(
        x[inlier_mask],
        y[inlier_mask],
        yerr=y_err[inlier_mask],
        fmt='none',
        ecolor='#ccccdd',
        elinewidth=0.5,
        zorder=2,
    )

    # plot the outliers as little arrows (w/ no error bars b/c they already are out of scale)
    if np.any(outlier_mask):
        outlier_x = x[outlier_mask]
        outlier_y = y[outlier_mask]
        outlier_scatter = plt.scatter(outlier_x, [y_max] * len(outlier_x), color="red", edgecolors="k", marker="^", s=40, zorder=4, clip_on=False)

        # add the text annotations (but they're way too clumped together so I guess w/e for now)
        if annotate_outliers:
            texts = []
            for x_val, actual_val in zip(outlier_x, outlier_y):
                t = ax.text(x_val, y_max*0.98, f"{actual_val:.1f}", ha='right', va='top', rotation=45, fontsize=8, zorder=5)
                texts.append(t)

            adjust_text(
                texts, 
                ax=ax, 
                only_move={'text': 'x'},        
                add_objects=[outlier_scatter],  
                expand_text=(0.1, 0.1),         
                arrowprops=dict(arrowstyle="-", color='gray', lw=0.5) 
            )

# set spacing to the closest multiple of a desired number, for aesthetics
def snap_x_ticks(ax, snap):
    auto_step = np.diff(ax.get_xticks())[0]
    new_step = max(snap, round(auto_step / snap) * snap)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(new_step))

def format_frames_to_mmss(x, _):
    secs_tot = int(x) // FPS
    mins = secs_tot // 60
    secs = secs_tot % 60
    return f"{mins:02d}:{secs:02d}"


def main(input_path):
    data = np.load(input_path)
    ax = plt.gca()

    x, y, y_err = produce_subsampled_xy(data, SUBSAMPLE_RATE)

    plot(ax, x, y, y_err, Y_MAX, ANNOTATE_OUTLIERS)

    plt.title("Video tracking error w.r.t. robot logs")
    plt.xlabel("time (minutes:seconds)")
    plt.ylabel("error (meters)")
    ax.set_ylim(0, Y_MAX)


    ax.set_axisbelow(True)
    plt.grid(True, linestyle="--")

    snap_x_ticks(ax, SNAP)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_frames_to_mmss))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y / 1000:.1f}"))

    plt.savefig(input_path.with_suffix(".eps"), format="eps") 
    plt.show()


data_path = Path(__file__).parent.parent / "data"
for p in [
    data_path / "2026-03-14_14-54-05_B-Human_HTWK-Robots/mario_A/error/by_frame_robotsonly__0.npy",
    data_path / "2026-03-14_14-54-05_B-Human_HTWK-Robots/mario_B/error/by_frame_robotsonly__1.npy",
    data_path / "2026-03-14_12-02-26_B-Human_HTWK-Robots__LARGEFINAL/mario_A/error/by_frame_robotsonly__0.npy"
]:
    main(p)
