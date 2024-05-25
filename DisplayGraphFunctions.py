import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

def plot_accuracy_with_error(data1, data2):
    # Function to prepare data
    def prepare_data(data):
        # Determine the maximum number of epochs in each dataset
        max_epochs = max(max(len(acc) for acc in data['accuracy']), max(len(val_acc) for val_acc in data['val_accuracy']))

        # Pad shorter lists with NaN
        padded_accuracy = [acc + [np.nan] * (max_epochs - len(acc)) for acc in data['accuracy']]
        padded_val_accuracy = [val_acc + [np.nan] * (max_epochs - len(val_acc)) for val_acc in data['val_accuracy']]

        # Convert lists to numpy arrays
        accuracy_array = np.array(padded_accuracy)
        val_accuracy_array = np.array(padded_val_accuracy)

        return accuracy_array, val_accuracy_array, max_epochs

    # Prepare data for both datasets
    accuracy_array1, val_accuracy_array1, max_epochs1 = prepare_data(data1)
    accuracy_array2, val_accuracy2, max_epochs2 = prepare_data(data2)

    # Concatenate the data arrays
    accuracy_array = np.hstack((accuracy_array1, accuracy_array2))
    val_accuracy_array = np.hstack((val_accuracy_array1, val_accuracy2))
    total_epochs = max_epochs1 + max_epochs2

    # Prepare data points
    epochs = np.arange(1, total_epochs + 1)
    accuracy_means = np.nanmean(accuracy_array, axis=0)
    val_accuracy_means = np.nanmean(val_accuracy_array, axis=0)
    accuracy_stds = np.nanstd(accuracy_array, axis=0)
    val_accuracy_stds = np.nanstd(val_accuracy_array, axis=0)

    # Plotting the data with error margins
    plt.figure(figsize=(12, 6))
    plt.errorbar(epochs[:max_epochs1], accuracy_means[:max_epochs1], yerr=accuracy_stds[:max_epochs1], label='Apmācības precizitāte', fmt='-o', color='blue', capsize=5)
    plt.errorbar(epochs[:max_epochs1], val_accuracy_means[:max_epochs1], yerr=val_accuracy_stds[:max_epochs1], label='Validācijas precizitāte', fmt='-o', color='red', capsize=5)
    plt.errorbar(epochs[max_epochs1:], accuracy_means[max_epochs1:], yerr=accuracy_stds[max_epochs1:], fmt='-o', color='blue', capsize=5, linestyle='--')
    plt.errorbar(epochs[max_epochs1:], val_accuracy_means[max_epochs1:], yerr=val_accuracy_stds[max_epochs1:], fmt='-o', color='red', capsize=5, linestyle='--')

    # Green line at the transition point
    plt.axvline(x=max_epochs1, color='green', linestyle='--', label='Sasldētā un atsaldētā slāņa maiņa')

    # Titles and labels
    plt.title('Modeļa apmācības un validācijas precizitāte')
    plt.xlabel('Epohi')
    plt.ylabel('Precizitāte')

    # Adjusting x-axis ticks
    if total_epochs > 20:
        tick_interval = max(1, total_epochs // 20)
        ticks = np.arange(1, total_epochs + 1, tick_interval)
        plt.xticks(ticks, labels=[str(int(tick)) for tick in ticks])
    else:
        plt.xticks(epochs)

    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def load_experiment_data(synData, denoising, seeds, repo_location, trainingRound):
    data = {
        'accuracy': [],
        'val_accuracy': []
    }
    
    # Iterate through each seed to load corresponding experiments
    for seed in seeds:
        if synData == 0:
            file_pattern = f"MobileNetV2_RD100_SD0_Seed{seed}_training_history_{trainingRound}.pkl"
        else:
            file_pattern = f"MobileNetV2_RD100_SD{synData}_denoising{denoising}_Seed{seed}_training_history_{trainingRound}.pkl"
        
        # Assuming there might be multiple experiments per seed
        
        file_path = os.path.join(repo_location, file_pattern)
        print(file_path)
        if not os.path.exists(file_path):
            print(f"No more files found at {file_path}, stopping.")
            break  # Stop if file does not exist
        
        # Load the experiment data from the file
        with open(file_path, 'rb') as file:
            history = pickle.load(file)
        
        # Append 'accuracy' and 'val_accuracy' data to the lists
        data['accuracy'].append(history.get('accuracy', []))
        data['val_accuracy'].append(history.get('val_accuracy', []))
            
            

    return data