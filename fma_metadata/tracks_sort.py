import pandas as pd
df = pd.read_csv('tracks_train.csv')

print(df.head())
print(df.columns.tolist())

licenses_to_delete =['Sampling Plus', 'Orphan Work','ideology.de', 'Art Libre','Free Music Philosophy', 'Electronic Frontier Foundation Open Audio License']

for l in df['license']:
	if pd.isna(l):
		continue
	name_lower = l.lower() 
	if "noncommercial" in name_lower or "non-commercial" in name_lower:
        	licenses_to_delete.append(l)
	if "sharealike" in name_lower or "share alike" in name_lower or "-sa" in name_lower:
		licenses_to_delete.append(l)
	if "pennsound" in name_lower or "no derivative" in name_lower or "noderivatives" in name_lower or "noderivs" in name_lower:
		licenses_to_delete.append(l)

# 2. Keep only rows where the license is NOT in that list
df = df[~df['license'].isin(licenses_to_delete)]

df["license"].value_counts().to_string("all_licenses.txt")
print("Saved all licenses to 'all_licenses.txt'")

df.to_csv('final_tracks.csv')
print(df['license'].value_counts())
