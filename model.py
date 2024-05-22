import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2M, MobileNetV3Small, MobileNetV2 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import TrainDataPrepare as td
import HelperFunctions as hf




pd.set_option('display.max_rows', 100)  # Increase the number of rows
pd.set_option('display.max_columns', 20)  # Increase the number of columns
pd.set_option('display.width', 2000)
pd.set_option('display.max_colwidth', None) 

def TrainOnMobileNetV2 (SIN_DIR, RANDOM_SEED, SYN_HALU = '0', SYNTHETIC_DATA_PERCENT = 0):
    PREPARE_DATA = True
    USE_SYNTHETIC_DATA = True
    SYNTHETIC_DATA_DIR = ''
    LOAD_ALREADY_MODEL = False

    images_train = 'prepared_data/train'
    images_val = 'prepared_data/val'
    images_test = 'prepared_data/test'
    train_csv_path = 'prepared_data/train_dataset.csv'
    val_csv_path = 'prepared_data/val_dataset.csv'
    test_csv_path = 'prepared_data/test_dataset.csv'
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 75
    REAL_DATA_PERCENT = 100
    #SYNTHETIC_DATA_PERCENT = 0
    #SIN_DIR = '../synthetic_data_0.15/train'
    MODEL_NAME = 'MobileNetV2'
    #SYN_HALU = '0.15'
    #RANDOM_SEED = 20

    folder_path = 'prepared_data'
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # Remove the folder
        hf.delete_directory(folder_path)


    if (PREPARE_DATA):
        #prepare csv
        #prepare data for traning and place them into prepared_data folder
        td.SelectDataByPercent('OOC_image_dataset/train', 'prepared_data/train', REAL_DATA_PERCENT, RANDOM_SEED)
        td.SelectDataByPercent(SIN_DIR, 'prepared_data/train', SYNTHETIC_DATA_PERCENT, RANDOM_SEED)

        #transform xlsx file to csv 
        hf.convert_xlsx_to_csv('OOC_datasheet.xlsx', 'OOC_datasheet.csv')

        #create csv file also for sythetic data 
        td.create_synthetic_csv_from_real_and_structure('OOC_datasheet.csv', SIN_DIR, 'OOC_datasheet_01.csv')

        #join both csv files toghether
        hf.concatenate_csv('OOC_datasheet.csv', 'OOC_datasheet_01.csv', 'OOC_datasheet_joined.csv')
        hf.delete_file('OOC_datasheet.csv')
        hf.delete_file('OOC_datasheet_01.csv')

        #transfer val and test datasets to prepared data
        hf.copy_directory('OOC_image_dataset/test/', 'prepared_data')
        hf.copy_directory('OOC_image_dataset/val/', 'prepared_data')

        #split the csv files into train, val, test
        hf.split_datasets_by_actual_images('OOC_datasheet_joined.csv', 'prepared_data', 'prepared_data')
        hf.delete_file('OOC_datasheet_joined.csv')

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)
    def collect_image_paths(dataset_type):
        paths = []
        for decision in ['good', 'bad']:
            decision_base_path = os.path.join('prepared_data', dataset_type, decision)
            for cell_type in os.listdir(decision_base_path):
                cell_type_path = os.path.join(decision_base_path, cell_type)
                if os.path.isdir(cell_type_path):
                    for day_dir in os.listdir(cell_type_path):
                        day_path = os.path.join(cell_type_path, day_dir)
                        if os.path.isdir(day_path):
                            for file in os.listdir(day_path):
                                if file.endswith('.png'):
                                    paths.append(os.path.join(day_path, file))
        return paths

    train_paths = collect_image_paths('train')
    val_paths = collect_image_paths('val')
    test_paths = collect_image_paths('test')

    def map_image_paths(df, image_paths):
        path_dict = {os.path.splitext(os.path.basename(path))[0]: path for path in image_paths}
        df['image_path'] = df['imageID'].map(path_dict)
        return df



    # Map the collected paths to the DataFrame based on the imageID
    train_df = map_image_paths(train_df, train_paths)
    val_df = map_image_paths(val_df, val_paths)
    test_df = map_image_paths(test_df, test_paths)

    train_df['label'] = train_df['Decision 1/2 (good/bad)'].map({1: 'Good', 2: 'Bad'})
    val_df['label'] = val_df['Decision 1/2 (good/bad)'].map({1: 'Good', 2: 'Bad'})
    test_df['label'] = test_df['Decision 1/2 (good/bad)'].map({1: 'Good', 2: 'Bad'})

    train_datagen = ImageDataGenerator(rescale = 1./255., vertical_flip = True, horizontal_flip = True)

    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator(rescale = 1./255.)

    # Flow training images in batches of 20 using train_datagen generator
    train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='image_path',
            y_col='label',
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=True,
            class_mode='binary'
        )  

    # Flow validation images in batches of 20 using test_datagen generator
    val_generator =  test_datagen.flow_from_dataframe(
            val_df,
            x_col='image_path',
            y_col='label',
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=False,
            class_mode='binary'
        )                          

    test_generator = test_datagen.flow_from_dataframe(
            test_df,
            x_col='image_path',
            y_col='label',
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=False,
            class_mode='binary'
    )

    # def plot_images(images, labels, num_images=10):
    #     plt.figure(figsize=(20, 10))
    #     for i in range(num_images):
    #         plt.subplot(2, 5, i + 1)
    #         plt.imshow(images[i])
    #         plt.title(str(labels[i]))
    #         plt.axis('off')
    #     plt.show()

    # # Retrieve one batch of data from the train, validation, or test generator
    # train_images, train_labels = next(train_generator)

    # # Assuming the data is structured as NumPy arrays and labels as numerical values or one-hot encodings
    # # If labels are one-hot encoded, convert to numerical labels
    # if train_labels.ndim > 1:
    #     train_labels = train_labels.argmax(axis=-1)

    # # Visualize 10 images from the batch
    # plot_images(train_images, train_labels, num_images=10)

    base_model = MobileNetV2(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), 
                                include_top=False,
                                weights='imagenet'
    )


    base_model.trainable = False
    #transfer learning un fine tuning

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # For binary classification
        #layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy'])
    #base model true

    early_stopping = EarlyStopping(
        monitor='val_accuracy',  # Metric to monitor, 'val_loss' or 'val_accuracy' are common options
        patience=10,          # Number of epochs to wait for an improvement before stopping
        restore_best_weights=True,  # Restore weights from the epoch with the best monitored metric
        verbose=1            # Verbosity level for logging the stopping action
    )

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=75,  # Set a higher initial value, since EarlyStopping will stop early if needed
        callbacks=[early_stopping]
    )

    import pickle
    saveLocation = ''
    if (SYN_HALU == '0'):
        saveLocation = 'results/' + MODEL_NAME + '_RD'  + str(REAL_DATA_PERCENT) + '_SD' + str(SYNTHETIC_DATA_PERCENT) + '_Seed' + str(RANDOM_SEED) + '_training_history_1.pkl'
    else:
        saveLocation = 'results/'+ MODEL_NAME + '_RD'  + str(REAL_DATA_PERCENT) + '_SD' + str(SYNTHETIC_DATA_PERCENT) + '_denoising' +SYN_HALU+ '_Seed' + str(RANDOM_SEED) +'_training_history_1.pkl'

    history_dict = history.history

    # Save the history.history dict to a pickle file
    with open(saveLocation, 'wb') as f:
        pickle.dump(history_dict, f)


    model.trainable = True

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00001),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=75,  # Set a higher initial value, since EarlyStopping will stop early if needed
        callbacks=[early_stopping]
    )


    # Step 5: Evaluation on test data
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"SinData: {SYNTHETIC_DATA_PERCENT}, SinDir: {SIN_DIR}, Seed: {RANDOM_SEED}, SynHalu: {SYN_HALU}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    import pickle
    saveLocation = ''
    if (SYN_HALU == '0'):
        saveLocation = 'results/' + MODEL_NAME + '_RD'  + str(REAL_DATA_PERCENT) + '_SD' + str(SYNTHETIC_DATA_PERCENT) + '_Seed' + str(RANDOM_SEED) + '_training_history_2.pkl'
    else:
        saveLocation = 'results/'+ MODEL_NAME + '_RD'  + str(REAL_DATA_PERCENT) + '_SD' + str(SYNTHETIC_DATA_PERCENT) + '_denoising' +SYN_HALU+ '_Seed' + str(RANDOM_SEED) +'_training_history_2.pkl'

    history_dict = history.history
    history_dict['test_loss'] = test_loss
    history_dict['test_acc'] = test_acc

    # Save the history.history dict to a pickle file
    with open(saveLocation, 'wb') as f:
        pickle.dump(history_dict, f)

    modelSave = ''
    if (SYN_HALU == '0'):
        modelSave = 'results/' +MODEL_NAME + '_RD'  + str(REAL_DATA_PERCENT) + '_SD' + str(SYNTHETIC_DATA_PERCENT) +'_Seed' + str(RANDOM_SEED) +'_modelSave.keras'
    else:
        modelSave = 'results/'+MODEL_NAME + '_RD'  + str(REAL_DATA_PERCENT) + '_SD' + str(SYNTHETIC_DATA_PERCENT) + '_denoising' +SYN_HALU+ '_Seed' + str(RANDOM_SEED) +'_modelSave.keras'
    model.save(modelSave)

    # import pickle

    # with open(saveLocation, 'rb') as f:
    #     loaded_history = pickle.load(f)
    # print(loaded_history)

    hf.delete_directory('prepared_data')

