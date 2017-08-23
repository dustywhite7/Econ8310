import pandas as pd
import numpy as np
from fbprophet import Prophet

# Prep the dataset

data = pd.read_csv("/home/dusty/Econ8310/DataSets/chicagoBusRiders.csv")
route3 = data[data.route=='3'][['date','rides']]
route3.date = pd.to_datetime(route3.date, infer_datetime_format=True)
route3.columns = [['ds', 'y']]

# Initialize Prophet instance and fit to data

m = Prophet()
m.fit(route3)

# Create timeline for 1 year in future, then generate predictions based on that timeline

future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

# Create plots of forecast and truth, as well as component breakdowns of the trends

plt = m.plot(forecast)
plt.show()

comp = m.plot_components(forecast)
comp.show()
