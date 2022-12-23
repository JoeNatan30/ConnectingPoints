import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

def video_path_format(path):
    '''
    - Delete any "NewLine Character" using .strip() 
    '''
    
    path = path.strip()

    return path

def is_banned_column(dataframe, pd_ban, column_name):

    pred = dataframe['Path'].isin(pd_ban)

    '''
    Notice the following:
     - True  means that is in the banned list - and banned instances is categorize by 0
     - False means that is a good instance - it is represent by 1 
    '''
    pred = pred.replace({True: 0, False: 1})

    matrix_ban = {'actual': [],
                  'predicted': []}

    matrix_ban['actual'] = list(dataframe[column_name])
    matrix_ban['predicted'] = list(pred)

    return matrix_ban

def plot_confusion_matrix(matrix_ban, name):

    title = name.replace('.png','').replace('_',' ')

    confusion_matrix = metrics.confusion_matrix(matrix_ban['actual'], matrix_ban['predicted'])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['outlier', 'inlier'])
    cm_display.plot()
    cm_display.ax_.set_title(title)
    plt.savefig(name)
    plt.close()  


diff = pd.read_excel('AEC_seleccionadas_total.xlsx',sheet_name=None)

diff_df = diff['glosses']

byVideoDur = pd.read_csv("../dataCleaningFunctions/banned_videos_by_duration.csv", header=None)
byVideoDur = byVideoDur[0].apply(video_path_format)
#print(byVideoDur)

byDistance = pd.read_csv("../dataCleaningFunctions/banned_videos_by_distance.csv", header=None)
byDistance = byDistance[0].apply(video_path_format)
#print(byDistance)

byBothBan = pd.concat([byVideoDur,byDistance])
byBothBan = byBothBan.drop_duplicates()
print(byBothBan)

diff_df['Path'] = diff_df['Path'].apply(video_path_format)

# Define prediction with zeros and ones | eliminate not checked instances
diff_df['¿Es o no es?'] = diff_df['¿Es o no es?'].replace('sí',1).replace('Sí',1)
diff_df['¿Es o no es?'] = diff_df['¿Es o no es?'].replace('no',0).replace('No',0)
diff_df = diff_df.dropna(subset=['¿Es o no es?'])
diff_df['¿Es o no es?'] = diff_df['¿Es o no es?'].astype(int)
print(diff_df[diff_df['¿Es o no es?'] == 0])
# BY DISTANCE
matrix_by_distance = is_banned_column(diff_df, byDistance, '¿Es o no es?')

# BY VIDEO DURATION
matrix_by_video_dur = is_banned_column(diff_df, byVideoDur, '¿Es o no es?')

# BY BOTH BAN METHOD
matrix_by_both_ban = is_banned_column(diff_df, byBothBan, '¿Es o no es?')

plot_confusion_matrix(matrix_by_distance, 'banned_by_distance.png')
plot_confusion_matrix(matrix_by_video_dur, 'banned_by_video_duration.png')
plot_confusion_matrix(matrix_by_both_ban, 'banned_by_both_method.png')