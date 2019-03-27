import os
from pydub import AudioSegment


def find_time_interval(wrd_file, keyword):
    time1 = ''
    time2 = ''

    with open(wrd_file, 'r') as f:
        for line in f:
            for word in line.split():
                if word == keyword:
                    return int(time1), int(time2)
                else:
                    time1 = time2
                    time2 = word


def extract_keyword_region(file_name, dir_name, keyword):
    wav_file_name = file_name
    title = os.path.splitext(wav_file_name)[0] # file name without extension
    wrd_file_name = title + '.WRD'
    wav_full_path = os.path.join(dir_name, wav_file_name)
    wrd_full_path = os.path.join(dir_name, wrd_file_name)

    (start_time, end_time) = find_time_interval(wrd_full_path, keyword)
    start_time = int (start_time * 1/16)
    end_time = int (end_time*1/16)

    new_audio = AudioSegment.from_wav(wav_full_path, "wav")
    new_audio = new_audio[start_time:end_time]

    return new_audio


def convert_all_files(source_path, destination_path):
    for dir_name, subdir_list, file_list in os.walk(source_path):
        for file_name in file_list:
            if file_name.endswith('.wav'):
                subdir_name = os.path.basename(dir_name) # keyword

                new_folder = os.path.join(destination_path, subdir_name)
                try:
                    os.mkdir(new_folder)
                except FileExistsError as e:
                    print('Folder %s already exists' % (subdir_name))

                new_full_path = os.path.join(new_folder, file_name)
                new_file = extract_keyword_region(file_name, dir_name, subdir_name)
                new_file.export(new_full_path, format='wav')


convert_all_files('/home/erika/Documents/Licenta/datasets/timit_full/TIMIT/data/lisa/data/timit/raw/TIMIT/Keywords',
                  '/home/erika/Documents/Licenta/datasets/timit_full/TIMIT/data/lisa/data/timit/raw/TIMIT/DATA/TRAIN'
                  '/Keywords')
