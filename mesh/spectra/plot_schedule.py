import json
import matplotlib.pyplot as plt

def plot_schedule(schedule_file='schedule.json'):
    """
    Generates and saves a Gantt chart visualization of the schedule.

    Reads a schedule from a JSON file and plots it using matplotlib,
    where the y-axis represents bin IDs, the x-axis represents time,
    and each scheduled job is a colored horizontal bar.

    Args:
        schedule_file (str): The path to the schedule JSON file.
    """
    try:
        with open(schedule_file, 'r') as f:
            schedule = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {schedule_file} was not found. Make sure run.jl executes first.")
        return

    fig, ax = plt.subplots(figsize=(15, 7))

    # Get all unique job IDs to create a consistent color map
    job_ids = sorted([int(j) for j in schedule.keys()])
    # Use a qualitative colormap for distinct colors
    colors = plt.cm.get_cmap('tab10', len(job_ids) if job_ids else 1)
    color_map = {job_id: colors(i) for i, job_id in enumerate(job_ids)}

    legend_handles = {}

    # schedule format: {job_id: {bin_id: [start, end]}}
    for job_id_str, bin_map in schedule.items():
        job_id = int(job_id_str)
        for bin_id_str, times in bin_map.items():
            bin_id = int(bin_id_str)
            start_time, end_time = times
            duration = end_time - start_time

            bar = ax.barh(
                y=bin_id, width=duration, left=start_time, height=0.5,
                align='center', color=color_map.get(job_id), edgecolor='black', linewidth=0.5
            )

            if job_id not in legend_handles:
                legend_handles[job_id] = bar

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Bin ID', fontsize=12)
    ax.set_title('Schedule Gantt Chart', fontsize=14)

    all_bins = {int(bin_id) for bin_map in schedule.values() for bin_id in bin_map.keys()}
    if all_bins:
        ax.set_yticks(sorted(list(all_bins)))

    ax.grid(axis='x', linestyle=':', alpha=0.7)
    ax.legend(
        [legend_handles[jid] for jid in sorted(legend_handles.keys())],
        [f'Permutation {jid}' for jid in sorted(legend_handles.keys())],
        title='Jobs', bbox_to_anchor=(1.02, 1), loc='upper left'
    )

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.show()

if __name__ == '__main__':
    plot_schedule()