# decomposition graph
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from pandas.plotting import register_matplotlib_converters
from pylab import rcParams

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10

df = pd.read_csv('../csv_files/main_dataset.csv', parse_dates=['timestamp'], index_col="timestamp",
                 infer_datetime_format=True)

decomposition = sm.tsa.seasonal_decompose(df['total_energy'])
fig, axes = plt.subplots(4, 1)

decomposition.observed.plot(ax=axes[0], legend=False)
axes[0].set_ylabel('Observed', fontSize='30')
decomposition.trend.plot(ax=axes[1], legend=False)
axes[1].set_ylabel('Trend', fontSize='30')
decomposition.seasonal.plot(ax=axes[2], legend=False)
axes[2].set_ylabel('Seasonal', fontSize='30')
decomposition.resid.plot(ax=axes[3], legend=False)
axes[3].set_ylabel('Residual', fontSize='30')

plt.xlabel('timestamp', fontSize='30')
plt.savefig('../assets/decomposition.png')
