import pandas as pd
df = pd.read_csv('tracks_train.csv')
 
print(df.head())
print(df.columns.tolist())

licenses_to_delete = [
    # Existing entries
    'CopyrightPlus',
    'ideology.de'
    'Attribution-NonCommercial',
    'Attribution-NonCommercial 3.0 International',
    'Attribution-Noncommercial-Share Alike 3.0 United States',
    'Attribution-NonCommercial-ShareAlike 3.0 International',
    'Attribution-NonCommercial-NoDerivatives (aka Music Sharing) 3.0 International',
    'Attribution-Noncommercial-No Derivative Works 3.0 United States',
    'Creative Commons Attribution-NonCommercial-NoDerivatives 4.0',
    'Attribution-NonCommercial-ShareAlike',
    'Attribution-Noncommercial 3.0 United States',
    
    # New additions from your list
    'Attribution-Noncommercial-Share Alike 2.5 UK: Scotland',
    'Attribution-NonCommercial-NoDerivs 2.5 Israel',
    'Noncommercial Sampling Plus',
    'Attribution-NonCommercial-NoDerivs 3.0 Italy',
    'Music Sharing',
    'Attribution-Noncommercial-NoDerivatives 2.0 Generic',
    'Attribution-NonCommercial-ShareAlike 3.0 Italy',
    'Attribution-Noncommercial-NoDerivatives 2.1 Japan',
    'Attribution-NonCommercial 2.5',
    'Attribution-Noncommercial-No Derivative Works 2.5 Mexico',
    'Attribution-NonCommercial 2.0 Chile',
    'Attribution-Noncommercial-Share Alike 2.0 Germany',
    'Attribution-Noncommercia-Share Alike 2.5 Mexico',
    'Attribution-Noncommercial-No Derivative Works 3.0 Spain',
    'Attribution-Noncommercial-No Derivative Works 3.0 Netherlands',
    'Attribution-NonCommercial 3.0 France',
    'Attribution-Noncommercial-NoDerivatives 2.5 Australia',
    'Attribution NonCommercial 2.5',
    'Attribution-Noncommercial 3.0 New Zealand',
    'Attribution-Noncommercial 2.0 France',
    'Attribution-NonCommercial-ShareAlike 3.0 France',
    'Attribution-NonCommercial-NoDerivs 3.0 Portugal',
    'Attribution-Noncommercial-No Derivative Works 2.5 Switzerland',
    'Attribution-Noncommercial-Share Alike 2.5 Israel',
    'Attribution-Noncommercial 2.5 Canada',
    'Attribution-Noncommercial 2.0 UK: England',
    'Attribution-Noncommercial-No Derivative Works 3.0 Croatia',
    'Attribution-Noncommercial-Share Alike 2.0 Generic',
    'Attribution-Noncommercial-No Derivative Works 2.5 Sweden',
    'Attribution-NonCommercial-NoDerivs 2.5 Colombia',
    'Attribution-NonCommercial-NoDerivs 3.0 Poland',
    'Attribution-Noncommercial-No Derivative Works 2.5 Denmark',
    'Attribution-Noncommercial-No Derivative Works 2.5 Portugal',
    'Attribution-Noncommercial 2.5 Brazil',
    'Attribution-Noncommercial 3.0 Austria',
    'Attribution-Noncommercial-NoDerivatives 2.0 Belgium'
]

# 2. Keep only rows where the license is NOT in that list
# (~ is the "NOT" operator, and .isin() checks the list)
df = df[~df['license'].isin(licenses_to_delete)]

df["license"].value_counts().to_string("all_licenses.txt")
print("Saved all 106 licenses to 'all_licenses.txt'")

print(df['license'].value_counts())
df:q
 index=False)
