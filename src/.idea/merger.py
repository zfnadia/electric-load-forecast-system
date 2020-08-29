import pandas as pd

df1 = pd.read_csv('D:\ThesisP2\out\Thesis_Data.csv')
# dataset = pd.read_csv('sample_data.csv',header=0,encoding = 'unicode_escape')

df2 = pd.read_csv("D:\ThesisP2\out\output.csv")
df3 = pd.read_csv("D:\ThesisP2\out\output1.csv")

final1 = df1.append(df2)
final = final1.append(df3)



final = final.drop(['Max. Demand (Sub-station end) MW', 'Rule Curve ft', 'Dhaka (Demand at Evening Peak)', 'Mymensingh (Demand at Evening Peak)', 'Chattogram (Demand at Evening Peak)'
                    , 'Sylhet (Demand at Evening Peak)', 'Khulna (Demand at Evening Peak)', 'Barishal (Demand at Evening Peak)',
                    'Rajshahi (Demand at Evening Peak)', 'Rangpur (Demand at Evening Peak)', 'Cumilla (Demand at Evening Peak)'], axis=1)

final.to_csv("final_dataset.csv", index=0)
print(final.head())
print(final.info())

# print(df2.head())