import os
import numpy as np


def find_string_in_file(file_text,keyword):
    for line in file_text:
        for word in line.split():
            if word == keyword:
                return True


def find_files_for_keyword(keyword, data_path_root):
    files = []
    for dir_name, subdir_list,file_list in os.walk(data_path_root):
        for file_name in file_list:
            if file_name.endswith(".TXT"):
                full_file_name = os.path.join(dir_name, file_name)
                with open(full_file_name,'r') as f:
                    if find_string_in_file(f,keyword):
                        files.append(full_file_name)

    return files


def find_files(keyword_list, data_path_root):
    keyword_file_list = []
    for keyword in keyword_list:
        file_list = find_files_for_keyword(keyword,data_path_root)
        keyword_file_list.append((list(file_list),file_list))
        print("%s: %d files" % (keyword, len(file_list)))

    return keyword_file_list


keywords = ['age', 'warm', 'year', 'money', 'children', 'development','problem', 'wash', 'lunch', 'water', 'house'
    , 'room', 'light', 'kitchen', 'time', 'hello', 'book', 'help', 'temperature', 'artists', 'beautiful', 'carry',
            'wash', 'review', 'breakdown', 'hostages', 'remind']
keywords_file_list = find_files(keywords,'/home/erika/Documents/Licenta/datasets/data/lisa/data/timit/raw/TIMIT')

print(len(keywords_file_list))



