#load the required packages
import pandas as pd
from Bio import SeqIO
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

# Ensure 'saved_models' directory exists
if not os.path.exists("new_saved_models_B"):
    os.makedirs("new_saved_models_B")

#load the targets file
target_file = pd.read_csv("target_file_new.csv")

#define the function for loading the fasta sequences
def load_fasta_geneid(filepath):
    sequences = {}
    #gene_ids=[]
    for record in SeqIO.parse(filepath, "fasta"):
        sequences[record.id] = str(record.seq)
        #gene_ids.append(record.id)
    return sequences
    
#define the function for one hot encoding
def onehot(seq):
    code = {'A': [1, 0, 0, 0],
            'C': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'T': [0, 0, 0, 1],
            'N':[0, 0, 0, 0]}
    encoded = np.zeros((len(seq), 4))
    for i, nt in enumerate(seq):
        if nt in ['A', 'C', 'G', 'T']:
            encoded[i, :] = code[nt]
        else:
            encoded[i, :] = code['N']
    return encoded
    
#load the fasta file
fasta_file_A = "gene_seq_A_embryo_f.txt"
fasta_file_B = "gene_seq_B.txt"

# Load sequences function
fasta_sequences_A = load_fasta_geneid(fasta_file_A)
fasta_sequences_B = load_fasta_geneid(fasta_file_B)

#create a dict of gene id and target values from the targets file
gene_target = target_file.set_index('gene_id')['targets'].to_dict()

#create empty lists to store the data
one_hot_encoded_seq_A = []  
targets_A = [] 
gene_ids_A = []

#one hot encode sequence and collect the targets
for gene_id, sequence in fasta_sequences_A.items():
    encoded_seq = onehot(sequence)
    if encoded_seq.shape == (3020, 4):
        one_hot_encoded_seq_A.append(encoded_seq)
        target_value = gene_target.get(gene_id, None)  # Get target value for each gene
        targets_A.append(target_value)
        gene_ids_A.append(gene_id)
    else:
        print(f"Skipping sequence for {gene_id}, shape: {encoded_seq.shape}")

print(len(gene_ids_A))
print(len(one_hot_encoded_seq_A))
print(len(targets_A))

#create empty lists to store the data
one_hot_encoded_seq_B = []  
targets_B = []
gene_ids_B = []

#one hot encode sequence and collect the targets
for gene_id, sequence in fasta_sequences_B.items():
    encoded_seq = onehot(sequence)
    if encoded_seq.shape == (3020, 4):
        one_hot_encoded_seq_B.append(encoded_seq)
        target_value = gene_target.get(gene_id, None)  # Get target value for each gene
        targets_B.append(target_value)
        gene_ids_B.append(gene_id)
    else:
        print(f"Skipping sequence for {gene_id}, shape: {encoded_seq.shape}")

print(len(gene_ids_B))
print(len(one_hot_encoded_seq_B))
print(len(targets_B))

        
def build_network(x_train, x_test, y_train, y_test, gene_ids_train, gene_ids_B):
    model = Sequential([
        # Conv Block 1
        Conv1D(64, kernel_size=8, activation='relu', padding='same', input_shape=(x_train.shape[1], x_train.shape[2])),
        Conv1D(64, kernel_size=8, activation='relu', padding='same'),
        MaxPooling1D(8, padding='same'),
        Dropout(0.25),

        # Conv Block 2
        Conv1D(128, kernel_size=8, activation='relu', padding='same'),
        Conv1D(128, kernel_size=8, activation='relu', padding='same'),
        MaxPooling1D(8, padding='same'),
        Dropout(0.25),

        # Conv Block 3
        Conv1D(64, kernel_size=8, activation='relu', padding='same'),
        Conv1D(64, kernel_size=8, activation='relu', padding='same'),
        MaxPooling1D(8, padding='same'),
        Dropout(0.25),

        # Fully connected Block
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification (0 or 1)
    ])
    print(model.summary())
    model_save_name= f'new_saved_models_B/new_model_split_B.keras'
    print(f"save path: {model_save_name}")
    model_chkpt = ModelCheckpoint(model_save_name, save_best_only=True, verbose=1)
    early_stop = EarlyStopping(patience=10, verbose=1)
    reduce_lr = ReduceLROnPlateau(patience=5, factor=0.1)
    
    #compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test), callbacks=[model_chkpt, early_stop, reduce_lr])
    print(f"Training completed.")
    
    # Load the best model saved during this split
    if os.path.exists(model_save_name):
        print(f"Model saved successfully for split: {model_save_name}")
        best_model = load_model(model_save_name)
        predictions = best_model.predict(x_test)
    else: 
        print(f"ModelCheckpoint did not save the model. Saving manually.")
        model.save(model_save_name)
        best_model = load_model(model_save_name)
        predictions = best_model.predict(x_test)
        
    # Evaluate performance
    if best_model:
        val_auroc = roc_auc_score(y_test, predictions)
        predictions_binary = predictions > 0.5 
        val_acc = accuracy_score(y_test, predictions_binary)
        val_precision = precision_score(y_test, predictions_binary)
        val_recall = recall_score(y_test, predictions_binary)
        val_f1 = f1_score(y_test, predictions_binary)
        conf_matrix = confusion_matrix(y_test, predictions_binary)
        
        metrics_data = {
            'Metric': ['Accuracy', 'AUROC', 'Precision', 'Recall', 'F1 Score', 'Confusion Matrix'],
            'Value': [f'{val_acc:.4f}', f'{val_auroc:.4f}', f'{val_precision:.4f}', f'{val_recall:.4f}', f'{val_f1:.4f}', conf_matrix]
        }
        
        # Print evaluation results for the current split
        #print(f"Test B - Accuracy: {val_acc:.4f}, AUROC: {val_auroc:.4f}")
        print("\nModel Performance Metrics:\n")
        print(metrics_data)
        print('---------------------------------------')
        #performance = [val_acc, val_auroc, x_train.shape[0]]
        performance = [val_acc, val_auroc, val_precision, val_recall, val_f1, conf_matrix, x_train.shape[0]]
        
        train_predictions_probabilities = best_model.predict(x_train)
        
        train_pred_df = pd.DataFrame({
            'Gene_ID': gene_ids_train,
            'True_Class': y_train.flatten(),
            'Prediction_Probability': train_predictions_probabilities.flatten()
        })
        
        test_pred_df = pd.DataFrame({
            'Gene_ID': gene_ids_B,
            'True_Class': y_test.flatten(),
            'Prediction_Probability': predictions.flatten()
        })
        
        # Save the prediction probabilities for both training and test sets to separate Excel files
        train_pred_df.to_csv('train_gene_predictions.csv', index=False)
        test_pred_df.to_csv('test_gene_predictions.csv', index=False)
        
        print("\nGene-level prediction probabilities saved.")
        return performance, train_pred_df, test_pred_df
    else:
    	print(f"Error: Model was not saved. Skipping!")
    	#return [None, None, x_train.shape[0]]
    	return [None, None, None, None, None, None, x_train.shape[0]]

def pre_process_train_test_split(x_train, x_test, y_train, y_test, gene_ids_A, gene_ids_B):
	final_training_output=[]
	#incomplete_splits = []
	
	#create a loop that 7 times(7A chromosomes)
	#for i in range(7):
	#	x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=i)
	#	print(f"Starting training for split {i+1}")
		
		# **Check Data Distribution for This Split**
		#print(f"Split {i+1}:")
	print(f"Train size: {len(x_train)}, Validation size: {len(x_test)}")
	print(f"Train class distribution: {pd.Series(y_train).value_counts(normalize=True)}")
	print(f"Validation class distribution: {pd.Series(y_test).value_counts(normalize=True)}")
		
		# Log unique indices for verification (Optional)
		#print(f"Train indices: {len(np.unique(x_train))}, Validation indices: {len(np.unique(x_val))}")
		# Check unique indices using pandas Series
		#train_indices = pd.Series(x_train).value_counts()
		#val_indices = pd.Series(x_val).value_counts()
		#print(f"Train indices: {len(train_indices)}, Validation indices: {len(val_indices)}")
		
		# Random Down-Sampling to Balance Data
		# Separate the classes (low_train = class 0, high_train = class 1)
	low_train, high_train = np.where(y_train == 0)[0], np.where(y_train == 1)[0]
	print(f"low_train: {len(low_train)}, high_train: {len(high_train)}")
		
	# Get the minimum class size and select that number from both classes
	min_class = min([len(low_train), len(high_train)])
	selected_low_train = np.random.choice(low_train, min_class, replace=False)
	selected_high_train = np.random.choice(high_train, min_class, replace=False)
		
	# Create balanced training data by concatenating the selected samples
	x_train = np.concatenate([np.take(x_train, selected_low_train, axis=0), np.take(x_train, selected_high_train, axis=0)], axis=0)
	y_train = np.concatenate([np.take(y_train, selected_low_train, axis=0), np.take(y_train, selected_high_train, axis=0)], axis=0)
	
	# Extract corresponding gene IDs for balanced training data
	gene_ids_train = np.concatenate([np.take(gene_ids_A, selected_low_train), np.take(gene_ids_A, selected_high_train)], axis=0)
		
	# Shuffle the balanced training data
	#x_train, y_train = shuffle(x_train, y_train, random_state=42)
		
	# **Verify Class Balancing**
	train_class_counts = pd.Series(y_train).value_counts()
	print(f"Balanced Train Class Distribution:\n{train_class_counts}")
	print(f"Gene_ids_train length:", len(gene_ids_train))
	print(f"Gene_ids_test length:", len(gene_ids_B))
	assert train_class_counts[0] == train_class_counts[1], "Error: Train classes not balanced!"
	try:
		# Build and train the model for this split
		performance = build_network(x_train, x_test, y_train, y_test, gene_ids_train, gene_ids_B)
		final_training_output.append(performance)
	except Exception as e:
		print(f"Error during training: {e}")
		#incomplete_splits.append(i+1)
		#continue
	# Convert results to DataFrame and display
	final_training_output = pd.DataFrame(final_training_output, columns=['val_acc', 'val_auROC', 'val_precision','val_recall', 'val_f1', 'conf_matrix', 'training size'])
	print("\nFinal Training Output\n", final_training_output.head())
	print("\nTraining Predictions\n", train_pred_df.head())
	print("\nTest Predictions\n", test_pred_df.head())
	# Save the DataFrame to a CSV file
	final_training_output.to_csv('final_training_output.csv', index=False)
	train_pred_df.to_csv('training_predictions.csv', index=False)
	test_pred_df.to_csv('test_predictions.csv', index=False
	
	#if incomplete_splits :
	#	print(f"\nWarning: Training was incomplete for splits: {incomplete_splits}")
	#else:
	#	print("\nAll splits completed successfully.")
	return final_training_output

x_train = np.array(one_hot_encoded_seq_A)
y_train = np.array(targets_A)
gene_ids_A = np.array(gene_ids_A)
x_test = np.array(one_hot_encoded_seq_B)
y_test = np.array(targets_B)
gene_ids_B = np.array(gene_ids_B)

results=pre_process_train_test_split(x_train, x_test, y_train, y_test, gene_ids_A, gene_ids_B)
this is the whole script i guess the error is near when we return the output to the performance 
