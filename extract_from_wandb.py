import wandb
import pandas as pd
import matplotlib.pyplot as plt

def extract_from_wandb():
    sweep_id = ''
    api = wandb.Api()
    sweep: wandb.sweep = api.sweep(sweep_id)
    runs = sweep.runs

    # Extract the data
    data = []
    for run in runs:
        summary = run.summary
        config = run.config
        data.append({
            'model_type': config['model_type'],
            'train_set_size': config['train_size'],
            'set_size': config['set_size'],
            'train loss': summary.get('train loss', None),
            'train accuracy': summary.get('train accuracy', None),
            'test loss': summary.get('test loss', None),
            'test accuracy': summary.get('test accuracy', None),
            'runtime': summary.get('_runtime', None),
        })

    # 1: Create plots showing accuracy vs. number of training examples for each
    # architecture, set size for both train and test data. it should be 4 plots: train/text x set_size1 / set_size2
    colors = ['r', 'b', 'y', 'm', 'c']
    df = pd.DataFrame(data)
    set_sizes = sorted(df['set_size'].unique(), reverse=False)
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))  # Adjust the size as needed
    fig.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust space between subplots
    axs = axs.flatten()

    for jdx, set_size in enumerate(set_sizes):
        for idx, metric in enumerate(['train accuracy', 'test accuracy']):
            ax = axs[jdx * 2 + idx]
            df_filtered = df[df['set_size'] == set_size]
            mean_metric = df_filtered.groupby(['train_set_size', 'model_type'])[metric].mean().reset_index()

            # Pivot the DataFrame to prepare for plotting
            pivot_df = mean_metric.pivot(index='train_set_size', columns='model_type', values=metric)

            # Plot the data
            for i, model_type in enumerate(pivot_df.columns):
                ax.plot(pivot_df.index, pivot_df[model_type], marker='o', color=colors[i],
                        label=f'Model Type = {model_type}')

            ax.set_xscale('log')
            ax.set_xlabel('Number of Training Examples')
            ax.set_ylabel(metric.capitalize().replace('_', ' '))
            ax.set_title(f'{metric.capitalize().replace("_", " ")} by Training Set Size\nSet Size = {set_size}')
            ax.legend(title='Model Type')
            ax.grid(True)

    # Save the entire figure
    plt.savefig('mean_metric_plot_grid.png')

    # Log the plot to WandB
    wandb.log({"Accuracy by Training Set Size and Set Size": wandb.Image('mean_metric_plot_grid.png')})

    # 2: Prepare a table summarizing the runtime of each architecture for both set
    # sizes and dimensionalities.


    # Replace NaN values with 'OOM' and convert runtime to minutes
    df['runtime'] = df['runtime'].apply(lambda x: 'OOM' if pd.isna(x) else f"{x / 60:.2f}")

    # Sort the DataFrame by train_set_size, set_size, and model_type in ascending order
    df_sorted = df.sort_values(by=['train_set_size', 'set_size', 'model_type'], ascending=[True, True, True])

    # Print the results in a table format
    print(f"{'Model':<10}  {'Runtime (min)':<15}")
    print("-" * 50)

    last_train_set_size = None
    last_set_size = None

    for index, row in df_sorted.iterrows():
        model = row['model_type'] or 'N/A'
        set_size = row['set_size'] or 'N/A'
        train_size = row['train_set_size'] or 'N/A'
        runtime = row['runtime'] or 'OOM'

        if last_train_set_size != train_size or last_set_size != set_size:
            # Print a separator if train_set_size or set_size changes
            if last_train_set_size is not None:
                print("-" * 50)
            print(f"\nTrain Set Size: {train_size} | Set Size: {set_size}\n")
            print(f"{'Model':<10} {'Runtime (min)':<15}")
            print("-" * 50)

        print(f"{model:<10} {runtime:<15}")

        last_train_set_size = train_size
        last_set_size = set_size