import os
import shutil

import argparse

parser = argparse.ArgumentParser(description='Get videos by glosses of a certain dataset')

parser.add_argument('--inpunt_file', default='AEC', type=str, help='path of the input file')
parser.add_argument('--output_file', default='to_check', type=str, help='path of the output file')
parser.add_argument('--words', default=[], nargs='+', help='list of words to collect')

args = parser.parse_args()

print()

inpunt_file = args.inpunt_file
group_of_words = args.words # ["abrir", "persona", "aburrido", "mucho", "sudar", "ver"]
out_path = args.output_file

num = 0
for folder, _, files in os.walk(inpunt_file):
    for file in files:
        word = file.split("_")[0].split("-")[0].lower()

        if word not in group_of_words:
            continue

        new_name = file.split('.')
        new_name = f'{new_name[0]}_{num}.{new_name[1]}'

        num = num + 1
        print(os.sep.join([folder,file]))
        print(os.sep.join([out_path,file]))
        shutil.copy2(os.sep.join([folder,file]), os.sep.join([out_path,new_name]))