from clearvoice import ClearVoice

##-----------------demo one: use one model ----------------------------------
if False:
    myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])

    ##1sd calling method: process an input waveform and return output waveform, then write to output.wav
    #output_wav = myClearVoice(input_path='input.wav', online_write=False)
    #myClearVoice.write(output_wav, output_path='output.wav')

    ##2nd calling method: process all wav files in 'path_to_input_wavs/' and write outputs to 'path_to_output_wavs'
    myClearVoice(input_path='path_to_input_wavs', online_write=True, output_path='path_to_output_wavs')

    ##3rd calling method: process wav files listed in .scp file, and write outputs to 'path_to_output_waves/'
    #myClearVoice(input_path='scp/cv_webrtc_test_set_20200521_16k.scp', online_write=True, output_path='path_to_output_scp')


##----------------Demo two: use multiple models -----------------------------------
if False:
    myClearVoice = ClearVoice(task='speech_enhancement', model_names=['FRCRN_SE_16K']) #, 'MossFormerGAN_SE_16K'])

    ##1sd calling method: process the waveform from input.wav and return output waveform, then write to output.wav
    #output_wav = myClearVoice(input_path='input.wav', online_write=False)
    #myClearVoice.write(output_wav, output_path='output.wav')

    ##2nd calling method: process all wav files in 'path_to_input_wavs/' and write outputs to 'path_to_output_wavs'
    myClearVoice(input_path='path_to_input_wavs', online_write=True, output_path='path_to_output_wavs')

    ##3rd calling method: process wav files listed in .scp file, and write outputs to 'path_to_output_waves/'
    #myClearVoice(input_path='scp/cv_webrtc_test_set_20200521_16k.scp', online_write=True, output_path='path_to_output_scp')

if False:
    myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormerGAN_SE_16K'])

    ##1sd calling method: process the waveform from input.wav and return output waveform, then write to output.wav
    #output_wav = myClearVoice(input_path='input.wav', online_write=False)
    #myClearVoice.write(output_wav, output_path='output.wav')

    ##2nd calling method: process all wav files in 'path_to_input_wavs/' and write outputs to 'path_to_output_wavs'
    myClearVoice(input_path='path_to_input_wavs', online_write=True, output_path='path_to_output_wavs')

    ##3rd calling method: process wav files listed in .scp file, and write outputs to 'path_to_output_waves/'
    #myClearVoice(input_path='scp/cv_webrtc_test_set_20200521_16k.scp', online_write=True, output_path='path_to_output_scp')

##----------------Demo three: use one model for speech separation -----------------------------------
if True:
    myClearVoice = ClearVoice(task='speech_separation', model_names=['MossFormer2_SS_16K'])

    ##1sd calling method: process an input waveform and return output waveform, then write to output.wav
    #output_wav = myClearVoice(input_path='input.wav', online_write=False)
    #myClearVoice.write(output_wav, output_path='output.wav')

    #2nd calling method: process all wav files in 'path_to_input_wavs/' and write outputs to 'path_to_output_wavs'
    #myClearVoice(input_path='path_to_input_wavs_ss', online_write=True, output_path='path_to_output_wavs')

    ##3rd calling method: process wav files listed in .scp file, and write outputs to 'path_to_output_waves/'
    myClearVoice(input_path='scp/libri_2mix_tt.scp', online_write=True, output_path='path_to_output_scp')

##----------------Demo four: use one model for audio-visual target speaker extraction -----------------------------------
if False:
    myClearVoice = ClearVoice(task='target_speaker_extraction', model_names=['AV_MossFormer2_TSE_16K'])

    # #1sd calling method: process an input video and return output video, then write outputs to 'path_to_output_videos_tse'
    # output_wav = myClearVoice(input_path='path_to_input_videos_tse/004.MOV', online_write=True, output_path='path_to_output_videos_tse')

    #2nd calling method: process all video files in 'path_to_input_videos/' and write outputs to 'path_to_output_videos_tse'
    myClearVoice(input_path='path_to_input_videos_tse', online_write=True, output_path='path_to_output_videos_tse')

    # #3rd calling method: process video files listed in .scp file, and write outputs to 'path_to_output_videos_tse/'
    # myClearVoice(input_path='scp/video_samples.scp', online_write=True, output_path='path_to_output_videos_tse')
