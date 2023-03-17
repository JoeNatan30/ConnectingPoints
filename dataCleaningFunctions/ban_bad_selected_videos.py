import pandas as pd


df = pd.read_excel(
     io="AEC_seleccionadas_total.xlsx",
     sheet_name="glosses"
)


df = df[df['¿Es o No es?']=='No']

print(df['Path'])
df['Path'].to_csv("banned_selected_videos.csv", header=False, index=False)
