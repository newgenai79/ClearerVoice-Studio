from clearvoice import ClearVoice  # Import the ClearVoice class for speech processing tasks

if __name__ == '__main__':
    ## ----------------- Demo One: Using a Single Model ----------------------
    if True:  # This block demonstrates how to use a single model for speech enhancement
        # Initialize ClearVoice for the task of speech enhancement using the MossFormerGAN_SE_16K model
        myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormerGAN_SE_16K'])

        # 1st calling method: 
        #   Process an input waveform and return the enhanced output waveform
        # - input_path: Path to the input noisy audio file (input.wav)
        # - The returned value is the enhanced output waveform
        output_wav = myClearVoice(input_path='input.wav')
        # Write the processed waveform to an output file
        # - output_wav: The enhanced waveform data
        # - output_path: Path to save the enhanced audio file (output.wav)
        myClearVoice.write(output_wav, output_path='output.wav')

        # 2nd calling method: 
        #   Process and write audio files directly
        # - input_path: Directory of input noisy audio files
        # - online_write=True: Enables writing the enhanced audio directly to files during processing
        # - output_path: Directory where the enhanced audio files will be saved
        myClearVoice(input_path='path_to_input_wavs', online_write=True, output_path='path_to_output_wavs')

        # 3rd calling method: 
        #   Use an .scp file to specify input audio paths
        # - input_path: Path to an .scp file listing multiple audio file paths
        # - online_write=True: Directly writes the enhanced output during processing
        # - output_path: Directory to save the enhanced output files
        myClearVoice(input_path='data/cv_webrtc_test_set_20200521_16k.scp', online_write=True, output_path='path_to_output_waves')


    ## ---------------- Demo Two: Using Multiple Models -----------------------
    if False:  # This block demonstrates using multiple models for speech enhancement
        # Initialize ClearVoice for the task of speech enhancement using two models: FRCRN_SE_16K and MossFormerGAN_SE_16K
        myClearVoice = ClearVoice(task='speech_enhancement', model_names=['FRCRN_SE_16K', 'MossFormerGAN_SE_16K'])

        # 1st calling method: 
        #   Process an input waveform using the multiple models and return the enhanced output waveform
        # - input_path: Path to the input noisy audio file (input.wav)
        # - The returned value is the enhanced output waveform after being processed by the models
        output_wav = myClearVoice(input_path='input.wav')
        # Write the processed waveform to an output file
        # - output_wav: The enhanced waveform data
        # - output_path: Path to save the enhanced audio file (output.wav)
        myClearVoice.write(output_wav, output_path='output.wav')

        # 2nd calling method: 
        #   Process and write audio files directly using multiple models
        # - input_path: Directory of input noisy audio files
        # - online_write=True: Enables writing the enhanced audio directly to files during processing
        # - output_path: Directory where the enhanced audio files will be saved
        myClearVoice(input_path='path_to_input_wavs', online_write=True, output_path='path_to_output_wavs')

        # 3rd calling method: 
        #   Use an .scp file to specify input audio paths for multiple models
        # - input_path: Path to an .scp file listing multiple audio file paths
        # - online_write=True: Directly writes the enhanced output during processing
        # - output_path: Directory to save the enhanced output files
        myClearVoice(input_path='data/cv_webrtc_test_set_20200521_16k.scp', online_write=True, output_path='path_to_output_waves')
