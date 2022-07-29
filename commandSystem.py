# Standard library imports
import os

# Third party imports

# Local imports
#

def change_options_state(chosen, ans):

    if chosen[ans]:
        chosen[ans] = False
    else:
        chosen[ans] = True

    return chosen

def get_selected_option_names(options,chosen):
    return [name for name, isChosed in zip(options, chosen.values()) if isChosed]

def show_options(options, name):

    print("##################################")
    print(f"Choose {name}: (by number)\n")

    for pos, opt in enumerate(options):
        print(f'{pos+1}) {opt}')

def select_datasets():

    options = os.listdir("./datasets")

    # delete options that are not dataset folders
    options = [opt for opt in options if len(opt.split('.'))==1]

    # dictionary of selected options
    chosen = {str(pos+1):False for pos in range(len(options))}

    while(True):

        show_options(options, 'datasets')

        print(f'\n{len(options)+1}) continue\n')

        chosen_names = get_selected_option_names(options,chosen)
        
        print("Selected:",*chosen_names)

        ans = input("write an option: ")

        if ans == str(len(options)+1):
            return chosen_names

        if ans not in chosen.keys():
            continue
        
        chosen = change_options_state(chosen, ans)


def select_keypoint_estimator():
    options = os.listdir("./keypointEstimators")

    # delete options that are not .py files 
    options = [opt.split('_')[0] for opt in options if opt.split('.')[-1]=='py']

    # dictionary of selected options
    chosen = {str(pos+1):False for pos in range(len(options))}

    while(True):

        show_options(options, 'keypoint estimators')

        print(f'\n{len(options)+1}) continue\n')

        chosen_names = get_selected_option_names(options,chosen)
        
        print("Selected:",*chosen_names)

        ans = input("write an option: ")

        if ans == str(len(options)+1):
            return chosen_names

        if ans not in chosen.keys():
            continue

        chosen = change_options_state(chosen, ans)