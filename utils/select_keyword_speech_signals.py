import os
import shutil
import numpy as np
import pylab as pl
from prettytable import PrettyTable
import random

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
                    if find_string_in_file(f, keyword):
                        files.append(full_file_name)

    return files


def find_files(keyword_list, data_path_root):
    keyword_file_list = []
    keyword_files_nr = []
    for keyword in keyword_list:
        file_list = find_files_for_keyword(keyword,data_path_root)
        keyword_file_list.append((list(file_list)))
        keyword_files_nr.append(len(file_list))
        print("%s: %d files" % (keyword, len(file_list)))
    keyword_files_nr = np.asarray(keyword_files_nr)
    return (keyword_file_list, keyword_files_nr)


def rename_and_move(source_path, destination_path):
    try:
        os.rename(source_path, destination_path)
    except FileNotFoundError as e:
        print('File %s not found\n' % (source_path))

def relative_select(length):
    rel_nr = 0
    if length < 20:
        rel_nr = 6
    elif length < 30:
        rel_nr = 10
    elif length < 50:
        rel_nr = 16
    else:
        rel_nr = 20

    return rel_nr


def select_files_to_relocate(keyword, files_found, extensions, destination_path, relative_selection = False, nr_files_relocate = 10):
    length  = len(files_found)
    if relative_selection:
        nr_relocate = relative_select(length)
    else:
        nr_relocate = nr_files_relocate

    for i in range(nr_relocate):
        print(len(files_found))
        random_file_path = random.choice(files_found)
        files_found.remove(random_file_path)
        file_name = os.path.basename(random_file_path)
        dir_name = os.path.dirname(random_file_path)
        extension = os.path.splitext(file_name)[1]
        title = os.path.splitext(file_name)[0]

        for ext in extensions:
            new_src_file_name = title+ext
            new_dest_file_name = title+'_' + str(i)+ext
            new_file_path = os.path.join(destination_path,keyword)
            try:
                os.mkdir(new_file_path)
            except FileExistsError as e:
                print('Folder %s already exists' % (keyword))
            new_file_path = os.path.join(new_file_path,new_dest_file_name)
            old_file_path = os.path.join(dir_name,new_src_file_name)
            rename_and_move(old_file_path,new_file_path)


def relocate_keyword_files(keywords, keywords_file_list, extensions,destination_path, relative_selection = False, nr_files_relocate = 6):
    length = len(keywords)
    for i in range(length):
        select_files_to_relocate(keywords[i], keywords_file_list[i], extensions, destination_path,
                                 relative_selection, nr_files_relocate=nr_files_relocate)


# testing file search for keywords
# keywords = ['money', 'children', 'water', 'house', 'time', 'carry', 'wash', 'problem', 'development', 'artists']
# indices = range(0, len(keywords))
# (keywords_file_list, files_nr) = find_files(keywords,'/home/erika/Documents/Licenta/datasets/timit_full/TIMIT/data/lisa'
#                                                      '/data/timit/raw/TIMIT/DATA/TRAIN')
# extensions = ['.TXT', '.wav', '.WRD', '.PHN', '.WAV']
#
# relocate_keyword_files(keywords, keywords_file_list, extensions, '/home/erika/Documents/Licenta/datasets/timit_full/'
#                                                                  'TIMIT/data/lisa/data/timit/raw/TIMIT/Keywords',
#                        relative_selection=True)

#
# print(len(keywords_file_list))
# pl.xticks(indices, keywords)
# pl.xticks(range(len(keywords)), keywords, rotation=45)
# pl.plot(indices, files_nr, '*')
# pl.show()
#
#
# table = PrettyTable()
# table.field_names = ['Keyword', 'Nr. of appearances']
#
# for i in indices:
#     table.add_row([keywords[i],files_nr[i]])
#
# print(table)
