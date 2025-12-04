1. Create a virtual environment using the command 'python -m venv venv'
2. Activate the virtual environment using the command 'venv\Scripts\activate'/'venv\Scripts\activate.ps1'
3. If you encounter any error, run this command in powershell to allow permission 'Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser' then follow step 2
4. Install all the dependencies listed in requirements.txt using the command 'pip install -r requirements.txt'
5. Go to the link specified in data/raw/link_to_houshold_power_consumption_data and download the .txt file and save it to data/raw folder
Note: specify the training set you want to train the model with, in each 'python' command specify either Training_Set-Up_1/src or Training_Set-Up_2/src
7. Preprocess the data using the command 'python src/preprocess_electricity.py --input data/raw/household_power_consumption.txt --out_dir data/processed/electricity --seq_len 168 --stride 24'
8. Start reconstruction pre-training using the command 'python src/train_recon_tf.py'
Note: The reconstruction training will be completed successfully but there will be an error while saving the reconstruction artifacts, ignore the error
9. Save reconstruction artifacts by using the command 'python src/save_recon_artifacts.py'
10. Start adversarial training by using the command 'python src/train_timegan_adversarial_tf.py'
11. Generate and save the synthetic data using the command 'python src/generate_and_save.py'
12. Evaluate the synthetic data by using the command 'python src/evaluate_synth_fixed.py'
13. Run jupyter notebooks for analysis
