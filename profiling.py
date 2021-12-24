import pandas as pd
from pandas_profiling import ProfileReport

data = pd.read_csv("data.csv", usecols=list(range(100))+[894])

profile = ProfileReport(data, title="Data Profiling Report")

profile.to_file("report.html")