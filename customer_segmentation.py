import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from yellowbrick.cluster import KElbowVisualizer
import warnings

pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 50)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
pd.set_option("display.width", 1000)
warnings.filterwarnings("ignore")


def check_data(dataframe, head=5):
    print ("####### SHAPE #######")
    print (dataframe.shape)
    print ("####### INFO #######")
    print (dataframe.info ())
    print ("####### DESCRIBE #######")
    print (dataframe.describe ([0.01, 0.1, 0.25, 0.50, 0.75, 0.9, 0.95, 0.99]))
    print ("####### NA VALUES #######")
    print (dataframe.isnull ().sum ())
    print ("####### FIRST {} ROWS #######".format (head))
    print (dataframe.head (head))


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Calculate lower and upper bounds for detecting outliers using the IQR (Interquartile Range) method.

    Parameters:
    -------
        dataframe (DataFrame): The DataFrame containing the data.
        col_name (str): The name of the column for which to calculate the outlier thresholds.
        q1 (float, optional): The lower quartile value (default is 0.25).
        q3 (float, optional): The upper quartile value (default is 0.75).

    Returns:
    ------
        float: The lower threshold for outliers.
        float: The upper threshold for outliers.
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95)
    if ( dataframe[ (dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit) ].any(axis =None) ):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe,variable,
                                              q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def select_country(dataframe, country):
   new_dataframe = dataframe.loc[dataframe["Country"]==country]
   return new_dataframe


def plot_histogram(dataframe, column_name, bins=20):
    """
    Plots a histogram of a specific column in the given DataFrame.
    
    Parameters:
        data (pandas.DataFrame): The DataFrame for which the histogram will be plotted.
        column_name (str): The name of the column for which the histogram will be plotted.
        num_bins (int, optional): The number of bins in the histogram. Default is 20.
    """
    dataframe[column_name].hist(bins=bins)
    plt.title(column_name)
    plt.show()

def segment_analysis_and_visualization(dataframe, segment_col, summary_args):
    """
    The function performs segment analysis and visualization on a given DataFrame using a specified
    segmentation column and summary statistics.

    Parameters:
        dataframe (pandas.DataFrame): The DataFrame containing the data.
        segment_col (str): The column that represents the segments created by the segmentation method.
        summary_args (dict): A dictionary specifying the summary statistics to be calculated for each feature.
    Returns:
        None (Displays analysis and visualizations).

    """
    dataframe.groupby(segment_col).agg(summary_args)
    print(f"Summary for {segment_col}:")
    plt.figure(figsize=(20, 10))
    sns.boxplot(x=segment_col, y="Monetary", data=dataframe)
    plt.title(f"Box Plot of Monetary by {segment_col}")
    plt.show()



df = pd.read_excel("dataset\online_retail_II-230817-120704.xlsx", sheet_name="Year 2009-2010")
# check_data(df)

#If Invoice code starts with 'C' it means the operation has been canceled
df = df[~df["Invoice"].str.contains("C", na=False)]

df = df[df["Quantity"] > 0]
df.dropna(inplace=True)

# print(df["StockCode"]=="M")

df = df[df["StockCode"] != 'M']

df["TotalPrice"] = df["Quantity"] * df["Price"]

# print(df[df["Price"] == 0]["StockCode"].unique ())

# Extracts unique StockCodes with at least three letters in the StockCode column.
invalid_codes = df[df["StockCode"].astype(str).str.contains(r"[a-zA-Z]{3,}")]["StockCode"].unique().tolist()

# print(invalid_codes)

#print(df[df["StockCode"].isin (invalid_codes)].groupby (["StockCode"]).agg ({"Invoice": "nunique",
#                                                                        "Quantity": "sum",
#                                                                        "Price": "sum",
#                                                                        "Customer ID": "nunique"}))
# ###***OUTPUT***###
#               Invoice  Quantity    Price  Customer ID
# StockCode
# ADJUST             32        32  3538.52           25
# ADJUST2             3         3   731.05            3
# BANK CHARGES       20        20   300.00           12
# PADS               14        14     0.01           12
# POST              738      2212 19964.83          230
# TEST001            11        60    40.50            4
# TEST002             1         1     1.00            1

df = df[~df["StockCode"].isin(invalid_codes)].reset_index(drop=True)

max_invoice_date = df["InvoiceDate"].max()
today_date= max_invoice_date + dt.timedelta(days=2)

rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda date: (today_date - date.max()).days,
                                     "Invoice":"nunique",
                                     "TotalPrice":"sum"})

print(rfm.head())

rfm.columns = ["Recency", "Frequency", "Monetary"]

rfm = rfm[(rfm["Frequency"] > 0) & (rfm["Monetary"] > 0)]

# for col in rfm.columns:
#     print(check_outlier(rfm, col))
for col in rfm.columns:
    replace_with_thresholds(rfm, col)


# plot_histogram(rfm, "Frequency")
# plot_histogram(rfm, "Recency")


# LOG Transformation
for col in ["Frequency", "Recency"]:
    rfm[f"LOG_{col}"] = np.log1p(rfm[col])

# plot_histogram(rfm, "LOG_Frequency")
# plot_histogram(rfm, "LOG_Recency")

# ****NORMALIZATION****

sc = StandardScaler()
sc.fit (rfm[["LOG_Recency", "LOG_Frequency"]])
scaled_rf = sc.transform (rfm[["LOG_Recency", "LOG_Frequency"]])

scaled_df = pd.DataFrame(index=rfm.index, columns=["LOG_Recency", "LOG_Frequency"], data=scaled_rf)
print(scaled_df.head())

# Determining optimal number of cluster

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=35)
elbow.fit(scaled_df)
elbow.show()
best_k = elbow.elbow_value_
print(best_k)

k_means = KMeans(n_clusters=best_k, ).fit(scaled_df)
labels = k_means.labels_

rfm["KMeansCluster"] = labels

print(rfm.tail(20))
print(rfm.shape)

args = {"Recency": ["mean", "median", "count"],
                    "Frequency":["mean", "median", "count"],
                    "Monetary": ["mean", "median", "count"]}


segment_analysis_and_visualization(rfm,"KmeansCluster", args)

### ***Hierarchical Clustering***

hc_complete = linkage(scaled_df,"complete")

plt.figure (figsize=(7, 5))
plt.title ("Dendrograms")
dend = dendrogram (hc_complete,
                   truncate_mode="lastp",
                   p=10,
                   show_contracted=True,
                   leaf_font_size=10)
plt.axhline (y=1.2, color='r', linestyle='--')
plt.show ()

hc = AgglomerativeClustering(n_clusters=6, )
segments= hc.fit_predict(scaled_df)
rfm["Hierarchical_Segments"] = segments

segment_analysis_and_visualization(rfm,"Hierarchical_Segments", args)
