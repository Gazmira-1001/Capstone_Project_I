---
jupyter:
  colab:
    name: Modelling+EDA_Group-7.ipynb
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown id="RCyeEm0OyHKm"}
# Data driven credit card fraud detection

Number total of transactions: 284,807 Number of frauds: 492 Number of
features: 31 (30 independents and 1 dependent) Unbalanced: fraud
represents only 0.172% of total transactions Only numeric input
variables: adapted by PCA transformation
:::

::: {.cell .markdown id="bxNef3EJyxqF"}
Importing the libraries
:::

::: {.cell .code execution_count="1" id="_sgc4taGJrTF"}
``` {.python}
import warnings 
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve,roc_auc_score, precision_recall_curve, average_precision_score

from termcolor import colored as cl # text customization
from sklearn.svm import SVC # SVM algorithm
from sklearn.metrics import accuracy_score # evaluation metric
from sklearn.metrics import f1_score # evaluation metric
from sklearn.preprocessing import StandardScaler, RobustScaler
import itertools # advanced tools

# Import the pipeline module we need for this from imblearn
from imblearn.pipeline import Pipeline 
from imblearn.over_sampling import BorderlineSMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
# for visualizing
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression
import matplotlib.patches as mpatches
from sklearn.ensemble import IsolationForest
```
:::

::: {.cell .markdown id="oq9voD4yy6LF"}
Connecting collab to drive
:::

::: {.cell .code execution_count="2" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="88AgGPleP30I" outputId="b1a4aca1-2a0d-4096-f829-dcbb2eebeb52"}
``` {.python}
from google.colab import drive
drive.mount('/content/drive')
```

::: {.output .stream .stdout}
    Mounted at /content/drive
:::
:::

::: {.cell .markdown id="0haMao2jzAHM"}
Importing data from drive
:::

::: {.cell .code execution_count="3" id="c0nhMHvhP0uI"}
``` {.python}
df = pd.read_csv('/content/drive/My Drive/creditcard.csv')
```
:::

::: {.cell .markdown id="NUFpXVqEzRBk"}
# EDA
:::

::: {.cell .markdown id="92YnhDDcP_m6"}
Shape of the data
:::

::: {.cell .code execution_count="8" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="5neUV9_8QIwB" outputId="9684c893-ee5c-405f-d7ec-5e082c6c505e"}
``` {.python}
df.shape
```

::: {.output .execute_result execution_count="8"}
    (284807, 31)
:::
:::

::: {.cell .markdown id="gWR0k6ZXQCGZ"}
Explanation of the datset
:::

::: {.cell .code execution_count="9" colab="{\"height\":393,\"base_uri\":\"https://localhost:8080/\"}" id="yBFagUUtQMbL" outputId="16779afa-996c-473e-b5b6-70c93c53a576"}
``` {.python}
df.describe()
```

::: {.output .execute_result execution_count="9"}
```{=html}
  <div id="df-c31d86fe-1472-4a85-80e9-7cbb18e427e3">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>284807.000000</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>...</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>284807.000000</td>
      <td>284807.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>94813.859575</td>
      <td>3.918649e-15</td>
      <td>5.682686e-16</td>
      <td>-8.761736e-15</td>
      <td>2.811118e-15</td>
      <td>-1.552103e-15</td>
      <td>2.040130e-15</td>
      <td>-1.698953e-15</td>
      <td>-1.893285e-16</td>
      <td>-3.147640e-15</td>
      <td>...</td>
      <td>1.473120e-16</td>
      <td>8.042109e-16</td>
      <td>5.282512e-16</td>
      <td>4.456271e-15</td>
      <td>1.426896e-15</td>
      <td>1.701640e-15</td>
      <td>-3.662252e-16</td>
      <td>-1.217809e-16</td>
      <td>88.349619</td>
      <td>0.001727</td>
    </tr>
    <tr>
      <th>std</th>
      <td>47488.145955</td>
      <td>1.958696e+00</td>
      <td>1.651309e+00</td>
      <td>1.516255e+00</td>
      <td>1.415869e+00</td>
      <td>1.380247e+00</td>
      <td>1.332271e+00</td>
      <td>1.237094e+00</td>
      <td>1.194353e+00</td>
      <td>1.098632e+00</td>
      <td>...</td>
      <td>7.345240e-01</td>
      <td>7.257016e-01</td>
      <td>6.244603e-01</td>
      <td>6.056471e-01</td>
      <td>5.212781e-01</td>
      <td>4.822270e-01</td>
      <td>4.036325e-01</td>
      <td>3.300833e-01</td>
      <td>250.120109</td>
      <td>0.041527</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-5.640751e+01</td>
      <td>-7.271573e+01</td>
      <td>-4.832559e+01</td>
      <td>-5.683171e+00</td>
      <td>-1.137433e+02</td>
      <td>-2.616051e+01</td>
      <td>-4.355724e+01</td>
      <td>-7.321672e+01</td>
      <td>-1.343407e+01</td>
      <td>...</td>
      <td>-3.483038e+01</td>
      <td>-1.093314e+01</td>
      <td>-4.480774e+01</td>
      <td>-2.836627e+00</td>
      <td>-1.029540e+01</td>
      <td>-2.604551e+00</td>
      <td>-2.256568e+01</td>
      <td>-1.543008e+01</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>54201.500000</td>
      <td>-9.203734e-01</td>
      <td>-5.985499e-01</td>
      <td>-8.903648e-01</td>
      <td>-8.486401e-01</td>
      <td>-6.915971e-01</td>
      <td>-7.682956e-01</td>
      <td>-5.540759e-01</td>
      <td>-2.086297e-01</td>
      <td>-6.430976e-01</td>
      <td>...</td>
      <td>-2.283949e-01</td>
      <td>-5.423504e-01</td>
      <td>-1.618463e-01</td>
      <td>-3.545861e-01</td>
      <td>-3.171451e-01</td>
      <td>-3.269839e-01</td>
      <td>-7.083953e-02</td>
      <td>-5.295979e-02</td>
      <td>5.600000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>84692.000000</td>
      <td>1.810880e-02</td>
      <td>6.548556e-02</td>
      <td>1.798463e-01</td>
      <td>-1.984653e-02</td>
      <td>-5.433583e-02</td>
      <td>-2.741871e-01</td>
      <td>4.010308e-02</td>
      <td>2.235804e-02</td>
      <td>-5.142873e-02</td>
      <td>...</td>
      <td>-2.945017e-02</td>
      <td>6.781943e-03</td>
      <td>-1.119293e-02</td>
      <td>4.097606e-02</td>
      <td>1.659350e-02</td>
      <td>-5.213911e-02</td>
      <td>1.342146e-03</td>
      <td>1.124383e-02</td>
      <td>22.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>139320.500000</td>
      <td>1.315642e+00</td>
      <td>8.037239e-01</td>
      <td>1.027196e+00</td>
      <td>7.433413e-01</td>
      <td>6.119264e-01</td>
      <td>3.985649e-01</td>
      <td>5.704361e-01</td>
      <td>3.273459e-01</td>
      <td>5.971390e-01</td>
      <td>...</td>
      <td>1.863772e-01</td>
      <td>5.285536e-01</td>
      <td>1.476421e-01</td>
      <td>4.395266e-01</td>
      <td>3.507156e-01</td>
      <td>2.409522e-01</td>
      <td>9.104512e-02</td>
      <td>7.827995e-02</td>
      <td>77.165000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>172792.000000</td>
      <td>2.454930e+00</td>
      <td>2.205773e+01</td>
      <td>9.382558e+00</td>
      <td>1.687534e+01</td>
      <td>3.480167e+01</td>
      <td>7.330163e+01</td>
      <td>1.205895e+02</td>
      <td>2.000721e+01</td>
      <td>1.559499e+01</td>
      <td>...</td>
      <td>2.720284e+01</td>
      <td>1.050309e+01</td>
      <td>2.252841e+01</td>
      <td>4.584549e+00</td>
      <td>7.519589e+00</td>
      <td>3.517346e+00</td>
      <td>3.161220e+01</td>
      <td>3.384781e+01</td>
      <td>25691.160000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 31 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c31d86fe-1472-4a85-80e9-7cbb18e427e3')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-c31d86fe-1472-4a85-80e9-7cbb18e427e3 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c31d86fe-1472-4a85-80e9-7cbb18e427e3');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code execution_count="10" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="7LxkZ7cqQSWK" outputId="8d0f21f4-1a6e-49bb-837d-e0b838a321c6"}
``` {.python}
print(df.isnull().any())
```

::: {.output .stream .stdout}
    Time      False
    V1        False
    V2        False
    V3        False
    V4        False
    V5        False
    V6        False
    V7        False
    V8        False
    V9        False
    V10       False
    V11       False
    V12       False
    V13       False
    V14       False
    V15       False
    V16       False
    V17       False
    V18       False
    V19       False
    V20       False
    V21       False
    V22       False
    V23       False
    V24       False
    V25       False
    V26       False
    V27       False
    V28       False
    Amount    False
    Class     False
    dtype: bool
:::
:::

::: {.cell .code colab="{\"height\":226,\"base_uri\":\"https://localhost:8080/\"}" id="JEpJ9KEJSNSd" outputId="51736f5a-af0d-4061-b3d3-0fdcc2df1e12"}
``` {.python}
df.head().style.set_properties(**{"background-color":"white",
                           "color" : "black"})
```

::: {.output .execute_result execution_count="126"}
```{=html}
<style type="text/css">
#T_d898a_row0_col0, #T_d898a_row0_col1, #T_d898a_row0_col2, #T_d898a_row0_col3, #T_d898a_row0_col4, #T_d898a_row0_col5, #T_d898a_row0_col6, #T_d898a_row0_col7, #T_d898a_row0_col8, #T_d898a_row0_col9, #T_d898a_row0_col10, #T_d898a_row0_col11, #T_d898a_row0_col12, #T_d898a_row0_col13, #T_d898a_row0_col14, #T_d898a_row0_col15, #T_d898a_row0_col16, #T_d898a_row0_col17, #T_d898a_row0_col18, #T_d898a_row0_col19, #T_d898a_row0_col20, #T_d898a_row0_col21, #T_d898a_row0_col22, #T_d898a_row0_col23, #T_d898a_row0_col24, #T_d898a_row0_col25, #T_d898a_row0_col26, #T_d898a_row0_col27, #T_d898a_row0_col28, #T_d898a_row0_col29, #T_d898a_row0_col30, #T_d898a_row1_col0, #T_d898a_row1_col1, #T_d898a_row1_col2, #T_d898a_row1_col3, #T_d898a_row1_col4, #T_d898a_row1_col5, #T_d898a_row1_col6, #T_d898a_row1_col7, #T_d898a_row1_col8, #T_d898a_row1_col9, #T_d898a_row1_col10, #T_d898a_row1_col11, #T_d898a_row1_col12, #T_d898a_row1_col13, #T_d898a_row1_col14, #T_d898a_row1_col15, #T_d898a_row1_col16, #T_d898a_row1_col17, #T_d898a_row1_col18, #T_d898a_row1_col19, #T_d898a_row1_col20, #T_d898a_row1_col21, #T_d898a_row1_col22, #T_d898a_row1_col23, #T_d898a_row1_col24, #T_d898a_row1_col25, #T_d898a_row1_col26, #T_d898a_row1_col27, #T_d898a_row1_col28, #T_d898a_row1_col29, #T_d898a_row1_col30, #T_d898a_row2_col0, #T_d898a_row2_col1, #T_d898a_row2_col2, #T_d898a_row2_col3, #T_d898a_row2_col4, #T_d898a_row2_col5, #T_d898a_row2_col6, #T_d898a_row2_col7, #T_d898a_row2_col8, #T_d898a_row2_col9, #T_d898a_row2_col10, #T_d898a_row2_col11, #T_d898a_row2_col12, #T_d898a_row2_col13, #T_d898a_row2_col14, #T_d898a_row2_col15, #T_d898a_row2_col16, #T_d898a_row2_col17, #T_d898a_row2_col18, #T_d898a_row2_col19, #T_d898a_row2_col20, #T_d898a_row2_col21, #T_d898a_row2_col22, #T_d898a_row2_col23, #T_d898a_row2_col24, #T_d898a_row2_col25, #T_d898a_row2_col26, #T_d898a_row2_col27, #T_d898a_row2_col28, #T_d898a_row2_col29, #T_d898a_row2_col30, #T_d898a_row3_col0, #T_d898a_row3_col1, #T_d898a_row3_col2, #T_d898a_row3_col3, #T_d898a_row3_col4, #T_d898a_row3_col5, #T_d898a_row3_col6, #T_d898a_row3_col7, #T_d898a_row3_col8, #T_d898a_row3_col9, #T_d898a_row3_col10, #T_d898a_row3_col11, #T_d898a_row3_col12, #T_d898a_row3_col13, #T_d898a_row3_col14, #T_d898a_row3_col15, #T_d898a_row3_col16, #T_d898a_row3_col17, #T_d898a_row3_col18, #T_d898a_row3_col19, #T_d898a_row3_col20, #T_d898a_row3_col21, #T_d898a_row3_col22, #T_d898a_row3_col23, #T_d898a_row3_col24, #T_d898a_row3_col25, #T_d898a_row3_col26, #T_d898a_row3_col27, #T_d898a_row3_col28, #T_d898a_row3_col29, #T_d898a_row3_col30, #T_d898a_row4_col0, #T_d898a_row4_col1, #T_d898a_row4_col2, #T_d898a_row4_col3, #T_d898a_row4_col4, #T_d898a_row4_col5, #T_d898a_row4_col6, #T_d898a_row4_col7, #T_d898a_row4_col8, #T_d898a_row4_col9, #T_d898a_row4_col10, #T_d898a_row4_col11, #T_d898a_row4_col12, #T_d898a_row4_col13, #T_d898a_row4_col14, #T_d898a_row4_col15, #T_d898a_row4_col16, #T_d898a_row4_col17, #T_d898a_row4_col18, #T_d898a_row4_col19, #T_d898a_row4_col20, #T_d898a_row4_col21, #T_d898a_row4_col22, #T_d898a_row4_col23, #T_d898a_row4_col24, #T_d898a_row4_col25, #T_d898a_row4_col26, #T_d898a_row4_col27, #T_d898a_row4_col28, #T_d898a_row4_col29, #T_d898a_row4_col30 {
  background-color: white;
  color: black;
}
</style>
<table id="T_d898a_" class="dataframe">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th class="col_heading level0 col0" >Time</th>
      <th class="col_heading level0 col1" >V1</th>
      <th class="col_heading level0 col2" >V2</th>
      <th class="col_heading level0 col3" >V3</th>
      <th class="col_heading level0 col4" >V4</th>
      <th class="col_heading level0 col5" >V5</th>
      <th class="col_heading level0 col6" >V6</th>
      <th class="col_heading level0 col7" >V7</th>
      <th class="col_heading level0 col8" >V8</th>
      <th class="col_heading level0 col9" >V9</th>
      <th class="col_heading level0 col10" >V10</th>
      <th class="col_heading level0 col11" >V11</th>
      <th class="col_heading level0 col12" >V12</th>
      <th class="col_heading level0 col13" >V13</th>
      <th class="col_heading level0 col14" >V14</th>
      <th class="col_heading level0 col15" >V15</th>
      <th class="col_heading level0 col16" >V16</th>
      <th class="col_heading level0 col17" >V17</th>
      <th class="col_heading level0 col18" >V18</th>
      <th class="col_heading level0 col19" >V19</th>
      <th class="col_heading level0 col20" >V20</th>
      <th class="col_heading level0 col21" >V21</th>
      <th class="col_heading level0 col22" >V22</th>
      <th class="col_heading level0 col23" >V23</th>
      <th class="col_heading level0 col24" >V24</th>
      <th class="col_heading level0 col25" >V25</th>
      <th class="col_heading level0 col26" >V26</th>
      <th class="col_heading level0 col27" >V27</th>
      <th class="col_heading level0 col28" >V28</th>
      <th class="col_heading level0 col29" >Amount</th>
      <th class="col_heading level0 col30" >Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_d898a_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_d898a_row0_col0" class="data row0 col0" >0.000000</td>
      <td id="T_d898a_row0_col1" class="data row0 col1" >-1.359807</td>
      <td id="T_d898a_row0_col2" class="data row0 col2" >-0.072781</td>
      <td id="T_d898a_row0_col3" class="data row0 col3" >2.536347</td>
      <td id="T_d898a_row0_col4" class="data row0 col4" >1.378155</td>
      <td id="T_d898a_row0_col5" class="data row0 col5" >-0.338321</td>
      <td id="T_d898a_row0_col6" class="data row0 col6" >0.462388</td>
      <td id="T_d898a_row0_col7" class="data row0 col7" >0.239599</td>
      <td id="T_d898a_row0_col8" class="data row0 col8" >0.098698</td>
      <td id="T_d898a_row0_col9" class="data row0 col9" >0.363787</td>
      <td id="T_d898a_row0_col10" class="data row0 col10" >0.090794</td>
      <td id="T_d898a_row0_col11" class="data row0 col11" >-0.551600</td>
      <td id="T_d898a_row0_col12" class="data row0 col12" >-0.617801</td>
      <td id="T_d898a_row0_col13" class="data row0 col13" >-0.991390</td>
      <td id="T_d898a_row0_col14" class="data row0 col14" >-0.311169</td>
      <td id="T_d898a_row0_col15" class="data row0 col15" >1.468177</td>
      <td id="T_d898a_row0_col16" class="data row0 col16" >-0.470401</td>
      <td id="T_d898a_row0_col17" class="data row0 col17" >0.207971</td>
      <td id="T_d898a_row0_col18" class="data row0 col18" >0.025791</td>
      <td id="T_d898a_row0_col19" class="data row0 col19" >0.403993</td>
      <td id="T_d898a_row0_col20" class="data row0 col20" >0.251412</td>
      <td id="T_d898a_row0_col21" class="data row0 col21" >-0.018307</td>
      <td id="T_d898a_row0_col22" class="data row0 col22" >0.277838</td>
      <td id="T_d898a_row0_col23" class="data row0 col23" >-0.110474</td>
      <td id="T_d898a_row0_col24" class="data row0 col24" >0.066928</td>
      <td id="T_d898a_row0_col25" class="data row0 col25" >0.128539</td>
      <td id="T_d898a_row0_col26" class="data row0 col26" >-0.189115</td>
      <td id="T_d898a_row0_col27" class="data row0 col27" >0.133558</td>
      <td id="T_d898a_row0_col28" class="data row0 col28" >-0.021053</td>
      <td id="T_d898a_row0_col29" class="data row0 col29" >149.620000</td>
      <td id="T_d898a_row0_col30" class="data row0 col30" >0</td>
    </tr>
    <tr>
      <th id="T_d898a_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_d898a_row1_col0" class="data row1 col0" >0.000000</td>
      <td id="T_d898a_row1_col1" class="data row1 col1" >1.191857</td>
      <td id="T_d898a_row1_col2" class="data row1 col2" >0.266151</td>
      <td id="T_d898a_row1_col3" class="data row1 col3" >0.166480</td>
      <td id="T_d898a_row1_col4" class="data row1 col4" >0.448154</td>
      <td id="T_d898a_row1_col5" class="data row1 col5" >0.060018</td>
      <td id="T_d898a_row1_col6" class="data row1 col6" >-0.082361</td>
      <td id="T_d898a_row1_col7" class="data row1 col7" >-0.078803</td>
      <td id="T_d898a_row1_col8" class="data row1 col8" >0.085102</td>
      <td id="T_d898a_row1_col9" class="data row1 col9" >-0.255425</td>
      <td id="T_d898a_row1_col10" class="data row1 col10" >-0.166974</td>
      <td id="T_d898a_row1_col11" class="data row1 col11" >1.612727</td>
      <td id="T_d898a_row1_col12" class="data row1 col12" >1.065235</td>
      <td id="T_d898a_row1_col13" class="data row1 col13" >0.489095</td>
      <td id="T_d898a_row1_col14" class="data row1 col14" >-0.143772</td>
      <td id="T_d898a_row1_col15" class="data row1 col15" >0.635558</td>
      <td id="T_d898a_row1_col16" class="data row1 col16" >0.463917</td>
      <td id="T_d898a_row1_col17" class="data row1 col17" >-0.114805</td>
      <td id="T_d898a_row1_col18" class="data row1 col18" >-0.183361</td>
      <td id="T_d898a_row1_col19" class="data row1 col19" >-0.145783</td>
      <td id="T_d898a_row1_col20" class="data row1 col20" >-0.069083</td>
      <td id="T_d898a_row1_col21" class="data row1 col21" >-0.225775</td>
      <td id="T_d898a_row1_col22" class="data row1 col22" >-0.638672</td>
      <td id="T_d898a_row1_col23" class="data row1 col23" >0.101288</td>
      <td id="T_d898a_row1_col24" class="data row1 col24" >-0.339846</td>
      <td id="T_d898a_row1_col25" class="data row1 col25" >0.167170</td>
      <td id="T_d898a_row1_col26" class="data row1 col26" >0.125895</td>
      <td id="T_d898a_row1_col27" class="data row1 col27" >-0.008983</td>
      <td id="T_d898a_row1_col28" class="data row1 col28" >0.014724</td>
      <td id="T_d898a_row1_col29" class="data row1 col29" >2.690000</td>
      <td id="T_d898a_row1_col30" class="data row1 col30" >0</td>
    </tr>
    <tr>
      <th id="T_d898a_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_d898a_row2_col0" class="data row2 col0" >1.000000</td>
      <td id="T_d898a_row2_col1" class="data row2 col1" >-1.358354</td>
      <td id="T_d898a_row2_col2" class="data row2 col2" >-1.340163</td>
      <td id="T_d898a_row2_col3" class="data row2 col3" >1.773209</td>
      <td id="T_d898a_row2_col4" class="data row2 col4" >0.379780</td>
      <td id="T_d898a_row2_col5" class="data row2 col5" >-0.503198</td>
      <td id="T_d898a_row2_col6" class="data row2 col6" >1.800499</td>
      <td id="T_d898a_row2_col7" class="data row2 col7" >0.791461</td>
      <td id="T_d898a_row2_col8" class="data row2 col8" >0.247676</td>
      <td id="T_d898a_row2_col9" class="data row2 col9" >-1.514654</td>
      <td id="T_d898a_row2_col10" class="data row2 col10" >0.207643</td>
      <td id="T_d898a_row2_col11" class="data row2 col11" >0.624501</td>
      <td id="T_d898a_row2_col12" class="data row2 col12" >0.066084</td>
      <td id="T_d898a_row2_col13" class="data row2 col13" >0.717293</td>
      <td id="T_d898a_row2_col14" class="data row2 col14" >-0.165946</td>
      <td id="T_d898a_row2_col15" class="data row2 col15" >2.345865</td>
      <td id="T_d898a_row2_col16" class="data row2 col16" >-2.890083</td>
      <td id="T_d898a_row2_col17" class="data row2 col17" >1.109969</td>
      <td id="T_d898a_row2_col18" class="data row2 col18" >-0.121359</td>
      <td id="T_d898a_row2_col19" class="data row2 col19" >-2.261857</td>
      <td id="T_d898a_row2_col20" class="data row2 col20" >0.524980</td>
      <td id="T_d898a_row2_col21" class="data row2 col21" >0.247998</td>
      <td id="T_d898a_row2_col22" class="data row2 col22" >0.771679</td>
      <td id="T_d898a_row2_col23" class="data row2 col23" >0.909412</td>
      <td id="T_d898a_row2_col24" class="data row2 col24" >-0.689281</td>
      <td id="T_d898a_row2_col25" class="data row2 col25" >-0.327642</td>
      <td id="T_d898a_row2_col26" class="data row2 col26" >-0.139097</td>
      <td id="T_d898a_row2_col27" class="data row2 col27" >-0.055353</td>
      <td id="T_d898a_row2_col28" class="data row2 col28" >-0.059752</td>
      <td id="T_d898a_row2_col29" class="data row2 col29" >378.660000</td>
      <td id="T_d898a_row2_col30" class="data row2 col30" >0</td>
    </tr>
    <tr>
      <th id="T_d898a_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_d898a_row3_col0" class="data row3 col0" >1.000000</td>
      <td id="T_d898a_row3_col1" class="data row3 col1" >-0.966272</td>
      <td id="T_d898a_row3_col2" class="data row3 col2" >-0.185226</td>
      <td id="T_d898a_row3_col3" class="data row3 col3" >1.792993</td>
      <td id="T_d898a_row3_col4" class="data row3 col4" >-0.863291</td>
      <td id="T_d898a_row3_col5" class="data row3 col5" >-0.010309</td>
      <td id="T_d898a_row3_col6" class="data row3 col6" >1.247203</td>
      <td id="T_d898a_row3_col7" class="data row3 col7" >0.237609</td>
      <td id="T_d898a_row3_col8" class="data row3 col8" >0.377436</td>
      <td id="T_d898a_row3_col9" class="data row3 col9" >-1.387024</td>
      <td id="T_d898a_row3_col10" class="data row3 col10" >-0.054952</td>
      <td id="T_d898a_row3_col11" class="data row3 col11" >-0.226487</td>
      <td id="T_d898a_row3_col12" class="data row3 col12" >0.178228</td>
      <td id="T_d898a_row3_col13" class="data row3 col13" >0.507757</td>
      <td id="T_d898a_row3_col14" class="data row3 col14" >-0.287924</td>
      <td id="T_d898a_row3_col15" class="data row3 col15" >-0.631418</td>
      <td id="T_d898a_row3_col16" class="data row3 col16" >-1.059647</td>
      <td id="T_d898a_row3_col17" class="data row3 col17" >-0.684093</td>
      <td id="T_d898a_row3_col18" class="data row3 col18" >1.965775</td>
      <td id="T_d898a_row3_col19" class="data row3 col19" >-1.232622</td>
      <td id="T_d898a_row3_col20" class="data row3 col20" >-0.208038</td>
      <td id="T_d898a_row3_col21" class="data row3 col21" >-0.108300</td>
      <td id="T_d898a_row3_col22" class="data row3 col22" >0.005274</td>
      <td id="T_d898a_row3_col23" class="data row3 col23" >-0.190321</td>
      <td id="T_d898a_row3_col24" class="data row3 col24" >-1.175575</td>
      <td id="T_d898a_row3_col25" class="data row3 col25" >0.647376</td>
      <td id="T_d898a_row3_col26" class="data row3 col26" >-0.221929</td>
      <td id="T_d898a_row3_col27" class="data row3 col27" >0.062723</td>
      <td id="T_d898a_row3_col28" class="data row3 col28" >0.061458</td>
      <td id="T_d898a_row3_col29" class="data row3 col29" >123.500000</td>
      <td id="T_d898a_row3_col30" class="data row3 col30" >0</td>
    </tr>
    <tr>
      <th id="T_d898a_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_d898a_row4_col0" class="data row4 col0" >2.000000</td>
      <td id="T_d898a_row4_col1" class="data row4 col1" >-1.158233</td>
      <td id="T_d898a_row4_col2" class="data row4 col2" >0.877737</td>
      <td id="T_d898a_row4_col3" class="data row4 col3" >1.548718</td>
      <td id="T_d898a_row4_col4" class="data row4 col4" >0.403034</td>
      <td id="T_d898a_row4_col5" class="data row4 col5" >-0.407193</td>
      <td id="T_d898a_row4_col6" class="data row4 col6" >0.095921</td>
      <td id="T_d898a_row4_col7" class="data row4 col7" >0.592941</td>
      <td id="T_d898a_row4_col8" class="data row4 col8" >-0.270533</td>
      <td id="T_d898a_row4_col9" class="data row4 col9" >0.817739</td>
      <td id="T_d898a_row4_col10" class="data row4 col10" >0.753074</td>
      <td id="T_d898a_row4_col11" class="data row4 col11" >-0.822843</td>
      <td id="T_d898a_row4_col12" class="data row4 col12" >0.538196</td>
      <td id="T_d898a_row4_col13" class="data row4 col13" >1.345852</td>
      <td id="T_d898a_row4_col14" class="data row4 col14" >-1.119670</td>
      <td id="T_d898a_row4_col15" class="data row4 col15" >0.175121</td>
      <td id="T_d898a_row4_col16" class="data row4 col16" >-0.451449</td>
      <td id="T_d898a_row4_col17" class="data row4 col17" >-0.237033</td>
      <td id="T_d898a_row4_col18" class="data row4 col18" >-0.038195</td>
      <td id="T_d898a_row4_col19" class="data row4 col19" >0.803487</td>
      <td id="T_d898a_row4_col20" class="data row4 col20" >0.408542</td>
      <td id="T_d898a_row4_col21" class="data row4 col21" >-0.009431</td>
      <td id="T_d898a_row4_col22" class="data row4 col22" >0.798278</td>
      <td id="T_d898a_row4_col23" class="data row4 col23" >-0.137458</td>
      <td id="T_d898a_row4_col24" class="data row4 col24" >0.141267</td>
      <td id="T_d898a_row4_col25" class="data row4 col25" >-0.206010</td>
      <td id="T_d898a_row4_col26" class="data row4 col26" >0.502292</td>
      <td id="T_d898a_row4_col27" class="data row4 col27" >0.219422</td>
      <td id="T_d898a_row4_col28" class="data row4 col28" >0.215153</td>
      <td id="T_d898a_row4_col29" class="data row4 col29" >69.990000</td>
      <td id="T_d898a_row4_col30" class="data row4 col30" >0</td>
    </tr>
  </tbody>
</table>
```
:::
:::

::: {.cell .code execution_count="19" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="dLgpP1AxS-Uz" outputId="fd4f5378-2742-4d7e-d2d0-048c40450fdb"}
``` {.python}
print(*df.columns, sep = ", ")
```

::: {.output .stream .stdout}
    Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount, Class
:::
:::

::: {.cell .code execution_count="20" colab="{\"height\":300,\"base_uri\":\"https://localhost:8080/\"}" id="A3jKdyfXTDDk" outputId="3eb712ae-d8e0-4d91-dfa2-c4877de764d9"}
``` {.python}
df[['Time', 'Amount']].describe().style.set_properties(**{"background-color":"white",
                           "color" : "black"})
```

::: {.output .execute_result execution_count="20"}
```{=html}
<style type="text/css">
#T_0b95f_row0_col0, #T_0b95f_row0_col1, #T_0b95f_row1_col0, #T_0b95f_row1_col1, #T_0b95f_row2_col0, #T_0b95f_row2_col1, #T_0b95f_row3_col0, #T_0b95f_row3_col1, #T_0b95f_row4_col0, #T_0b95f_row4_col1, #T_0b95f_row5_col0, #T_0b95f_row5_col1, #T_0b95f_row6_col0, #T_0b95f_row6_col1, #T_0b95f_row7_col0, #T_0b95f_row7_col1 {
  background-color: white;
  color: black;
}
</style>
<table id="T_0b95f_" class="dataframe">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th class="col_heading level0 col0" >Time</th>
      <th class="col_heading level0 col1" >Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_0b95f_level0_row0" class="row_heading level0 row0" >count</th>
      <td id="T_0b95f_row0_col0" class="data row0 col0" >284807.000000</td>
      <td id="T_0b95f_row0_col1" class="data row0 col1" >284807.000000</td>
    </tr>
    <tr>
      <th id="T_0b95f_level0_row1" class="row_heading level0 row1" >mean</th>
      <td id="T_0b95f_row1_col0" class="data row1 col0" >94813.859575</td>
      <td id="T_0b95f_row1_col1" class="data row1 col1" >88.349619</td>
    </tr>
    <tr>
      <th id="T_0b95f_level0_row2" class="row_heading level0 row2" >std</th>
      <td id="T_0b95f_row2_col0" class="data row2 col0" >47488.145955</td>
      <td id="T_0b95f_row2_col1" class="data row2 col1" >250.120109</td>
    </tr>
    <tr>
      <th id="T_0b95f_level0_row3" class="row_heading level0 row3" >min</th>
      <td id="T_0b95f_row3_col0" class="data row3 col0" >0.000000</td>
      <td id="T_0b95f_row3_col1" class="data row3 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_0b95f_level0_row4" class="row_heading level0 row4" >25%</th>
      <td id="T_0b95f_row4_col0" class="data row4 col0" >54201.500000</td>
      <td id="T_0b95f_row4_col1" class="data row4 col1" >5.600000</td>
    </tr>
    <tr>
      <th id="T_0b95f_level0_row5" class="row_heading level0 row5" >50%</th>
      <td id="T_0b95f_row5_col0" class="data row5 col0" >84692.000000</td>
      <td id="T_0b95f_row5_col1" class="data row5 col1" >22.000000</td>
    </tr>
    <tr>
      <th id="T_0b95f_level0_row6" class="row_heading level0 row6" >75%</th>
      <td id="T_0b95f_row6_col0" class="data row6 col0" >139320.500000</td>
      <td id="T_0b95f_row6_col1" class="data row6 col1" >77.165000</td>
    </tr>
    <tr>
      <th id="T_0b95f_level0_row7" class="row_heading level0 row7" >max</th>
      <td id="T_0b95f_row7_col0" class="data row7 col0" >172792.000000</td>
      <td id="T_0b95f_row7_col1" class="data row7 col1" >25691.160000</td>
    </tr>
  </tbody>
</table>
```
:::
:::

::: {.cell .code execution_count="22" colab="{\"height\":1000,\"base_uri\":\"https://localhost:8080/\"}" id="7BAUuHrWTT_g" outputId="1e1638f5-ea91-488c-f16d-d210c269772e"}
``` {.python}
# Descriptive statistics of  of frauds transactions
#Mean transaction is around 122 and standard deviation is around 256.
# Maximum Transaction was 2125 and minimum was 0.
summary = (df[df['Class'] == 1].describe().transpose().reset_index())
summary = summary.rename(columns = {"index" : "feature"})
summary = np.around(summary,3)

val_lst = [summary['feature'], summary['count'],
           summary['mean'],summary['std'],
           summary['min'], summary['25%'],
           summary['50%'], summary['75%'], summary['max']]

summary
```

::: {.output .execute_result execution_count="22"}
```{=html}
  <div id="df-d956d8cd-6daf-4f08-8330-9b13b500c3c5">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Time</td>
      <td>492.0</td>
      <td>80746.807</td>
      <td>47835.365</td>
      <td>406.000</td>
      <td>41241.500</td>
      <td>75568.500</td>
      <td>128483.000</td>
      <td>170348.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>V1</td>
      <td>492.0</td>
      <td>-4.772</td>
      <td>6.784</td>
      <td>-30.552</td>
      <td>-6.036</td>
      <td>-2.342</td>
      <td>-0.419</td>
      <td>2.132</td>
    </tr>
    <tr>
      <th>2</th>
      <td>V2</td>
      <td>492.0</td>
      <td>3.624</td>
      <td>4.291</td>
      <td>-8.402</td>
      <td>1.188</td>
      <td>2.718</td>
      <td>4.971</td>
      <td>22.058</td>
    </tr>
    <tr>
      <th>3</th>
      <td>V3</td>
      <td>492.0</td>
      <td>-7.033</td>
      <td>7.111</td>
      <td>-31.104</td>
      <td>-8.643</td>
      <td>-5.075</td>
      <td>-2.276</td>
      <td>2.250</td>
    </tr>
    <tr>
      <th>4</th>
      <td>V4</td>
      <td>492.0</td>
      <td>4.542</td>
      <td>2.873</td>
      <td>-1.313</td>
      <td>2.373</td>
      <td>4.177</td>
      <td>6.349</td>
      <td>12.115</td>
    </tr>
    <tr>
      <th>5</th>
      <td>V5</td>
      <td>492.0</td>
      <td>-3.151</td>
      <td>5.372</td>
      <td>-22.106</td>
      <td>-4.793</td>
      <td>-1.523</td>
      <td>0.215</td>
      <td>11.095</td>
    </tr>
    <tr>
      <th>6</th>
      <td>V6</td>
      <td>492.0</td>
      <td>-1.398</td>
      <td>1.858</td>
      <td>-6.406</td>
      <td>-2.502</td>
      <td>-1.425</td>
      <td>-0.413</td>
      <td>6.474</td>
    </tr>
    <tr>
      <th>7</th>
      <td>V7</td>
      <td>492.0</td>
      <td>-5.569</td>
      <td>7.207</td>
      <td>-43.557</td>
      <td>-7.965</td>
      <td>-3.034</td>
      <td>-0.946</td>
      <td>5.803</td>
    </tr>
    <tr>
      <th>8</th>
      <td>V8</td>
      <td>492.0</td>
      <td>0.571</td>
      <td>6.798</td>
      <td>-41.044</td>
      <td>-0.195</td>
      <td>0.622</td>
      <td>1.765</td>
      <td>20.007</td>
    </tr>
    <tr>
      <th>9</th>
      <td>V9</td>
      <td>492.0</td>
      <td>-2.581</td>
      <td>2.501</td>
      <td>-13.434</td>
      <td>-3.872</td>
      <td>-2.209</td>
      <td>-0.788</td>
      <td>3.354</td>
    </tr>
    <tr>
      <th>10</th>
      <td>V10</td>
      <td>492.0</td>
      <td>-5.677</td>
      <td>4.897</td>
      <td>-24.588</td>
      <td>-7.757</td>
      <td>-4.579</td>
      <td>-2.614</td>
      <td>4.031</td>
    </tr>
    <tr>
      <th>11</th>
      <td>V11</td>
      <td>492.0</td>
      <td>3.800</td>
      <td>2.679</td>
      <td>-1.702</td>
      <td>1.973</td>
      <td>3.586</td>
      <td>5.307</td>
      <td>12.019</td>
    </tr>
    <tr>
      <th>12</th>
      <td>V12</td>
      <td>492.0</td>
      <td>-6.259</td>
      <td>4.654</td>
      <td>-18.684</td>
      <td>-8.688</td>
      <td>-5.503</td>
      <td>-2.974</td>
      <td>1.376</td>
    </tr>
    <tr>
      <th>13</th>
      <td>V13</td>
      <td>492.0</td>
      <td>-0.109</td>
      <td>1.105</td>
      <td>-3.128</td>
      <td>-0.979</td>
      <td>-0.066</td>
      <td>0.673</td>
      <td>2.815</td>
    </tr>
    <tr>
      <th>14</th>
      <td>V14</td>
      <td>492.0</td>
      <td>-6.972</td>
      <td>4.279</td>
      <td>-19.214</td>
      <td>-9.693</td>
      <td>-6.730</td>
      <td>-4.283</td>
      <td>3.442</td>
    </tr>
    <tr>
      <th>15</th>
      <td>V15</td>
      <td>492.0</td>
      <td>-0.093</td>
      <td>1.050</td>
      <td>-4.499</td>
      <td>-0.644</td>
      <td>-0.057</td>
      <td>0.609</td>
      <td>2.471</td>
    </tr>
    <tr>
      <th>16</th>
      <td>V16</td>
      <td>492.0</td>
      <td>-4.140</td>
      <td>3.865</td>
      <td>-14.130</td>
      <td>-6.563</td>
      <td>-3.550</td>
      <td>-1.226</td>
      <td>3.140</td>
    </tr>
    <tr>
      <th>17</th>
      <td>V17</td>
      <td>492.0</td>
      <td>-6.666</td>
      <td>6.971</td>
      <td>-25.163</td>
      <td>-11.945</td>
      <td>-5.303</td>
      <td>-1.342</td>
      <td>6.739</td>
    </tr>
    <tr>
      <th>18</th>
      <td>V18</td>
      <td>492.0</td>
      <td>-2.246</td>
      <td>2.899</td>
      <td>-9.499</td>
      <td>-4.665</td>
      <td>-1.664</td>
      <td>0.092</td>
      <td>3.790</td>
    </tr>
    <tr>
      <th>19</th>
      <td>V19</td>
      <td>492.0</td>
      <td>0.681</td>
      <td>1.540</td>
      <td>-3.682</td>
      <td>-0.299</td>
      <td>0.647</td>
      <td>1.649</td>
      <td>5.228</td>
    </tr>
    <tr>
      <th>20</th>
      <td>V20</td>
      <td>492.0</td>
      <td>0.372</td>
      <td>1.347</td>
      <td>-4.128</td>
      <td>-0.172</td>
      <td>0.285</td>
      <td>0.822</td>
      <td>11.059</td>
    </tr>
    <tr>
      <th>21</th>
      <td>V21</td>
      <td>492.0</td>
      <td>0.714</td>
      <td>3.869</td>
      <td>-22.798</td>
      <td>0.042</td>
      <td>0.592</td>
      <td>1.245</td>
      <td>27.203</td>
    </tr>
    <tr>
      <th>22</th>
      <td>V22</td>
      <td>492.0</td>
      <td>0.014</td>
      <td>1.495</td>
      <td>-8.887</td>
      <td>-0.534</td>
      <td>0.048</td>
      <td>0.617</td>
      <td>8.362</td>
    </tr>
    <tr>
      <th>23</th>
      <td>V23</td>
      <td>492.0</td>
      <td>-0.040</td>
      <td>1.580</td>
      <td>-19.254</td>
      <td>-0.342</td>
      <td>-0.073</td>
      <td>0.308</td>
      <td>5.466</td>
    </tr>
    <tr>
      <th>24</th>
      <td>V24</td>
      <td>492.0</td>
      <td>-0.105</td>
      <td>0.516</td>
      <td>-2.028</td>
      <td>-0.437</td>
      <td>-0.061</td>
      <td>0.285</td>
      <td>1.091</td>
    </tr>
    <tr>
      <th>25</th>
      <td>V25</td>
      <td>492.0</td>
      <td>0.041</td>
      <td>0.797</td>
      <td>-4.782</td>
      <td>-0.314</td>
      <td>0.088</td>
      <td>0.457</td>
      <td>2.208</td>
    </tr>
    <tr>
      <th>26</th>
      <td>V26</td>
      <td>492.0</td>
      <td>0.052</td>
      <td>0.472</td>
      <td>-1.153</td>
      <td>-0.259</td>
      <td>0.004</td>
      <td>0.397</td>
      <td>2.745</td>
    </tr>
    <tr>
      <th>27</th>
      <td>V27</td>
      <td>492.0</td>
      <td>0.171</td>
      <td>1.377</td>
      <td>-7.263</td>
      <td>-0.020</td>
      <td>0.395</td>
      <td>0.826</td>
      <td>3.052</td>
    </tr>
    <tr>
      <th>28</th>
      <td>V28</td>
      <td>492.0</td>
      <td>0.076</td>
      <td>0.547</td>
      <td>-1.869</td>
      <td>-0.109</td>
      <td>0.146</td>
      <td>0.381</td>
      <td>1.779</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Amount</td>
      <td>492.0</td>
      <td>122.211</td>
      <td>256.683</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>9.250</td>
      <td>105.890</td>
      <td>2125.870</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Class</td>
      <td>492.0</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d956d8cd-6daf-4f08-8330-9b13b500c3c5')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-d956d8cd-6daf-4f08-8330-9b13b500c3c5 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d956d8cd-6daf-4f08-8330-9b13b500c3c5');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code execution_count="23" colab="{\"height\":1000,\"base_uri\":\"https://localhost:8080/\"}" id="OHCdXK-NYz_B" outputId="9f3aaaeb-a2a3-49c8-a002-969b94263ba9"}
``` {.python}
#Mean transaction is around 88 and standard deviation is around 250.
# Maximum Transaction was 25691 and minimum was 0.
summary = (df[df['Class'] == 0].describe().transpose().reset_index())
summary = summary.rename(columns = {"index" : "feature"})
summary = np.around(summary,3)
val_lst = [summary['feature'], summary['count'],
           summary['mean'],summary['std'],
           summary['min'], summary['25%'],
           summary['50%'], summary['75%'], summary['max']]
print("Summary for NonFraud Transactions")
summary
```

::: {.output .stream .stdout}
    Summary for NonFraud Transactions
:::

::: {.output .execute_result execution_count="23"}
```{=html}
  <div id="df-5483646c-831b-4623-a15b-2fcc096086e4">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Time</td>
      <td>284315.0</td>
      <td>94838.202</td>
      <td>47484.016</td>
      <td>0.000</td>
      <td>54230.000</td>
      <td>84711.000</td>
      <td>139333.000</td>
      <td>172792.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>V1</td>
      <td>284315.0</td>
      <td>0.008</td>
      <td>1.930</td>
      <td>-56.408</td>
      <td>-0.918</td>
      <td>0.020</td>
      <td>1.316</td>
      <td>2.455</td>
    </tr>
    <tr>
      <th>2</th>
      <td>V2</td>
      <td>284315.0</td>
      <td>-0.006</td>
      <td>1.636</td>
      <td>-72.716</td>
      <td>-0.599</td>
      <td>0.064</td>
      <td>0.800</td>
      <td>18.902</td>
    </tr>
    <tr>
      <th>3</th>
      <td>V3</td>
      <td>284315.0</td>
      <td>0.012</td>
      <td>1.459</td>
      <td>-48.326</td>
      <td>-0.885</td>
      <td>0.182</td>
      <td>1.028</td>
      <td>9.383</td>
    </tr>
    <tr>
      <th>4</th>
      <td>V4</td>
      <td>284315.0</td>
      <td>-0.008</td>
      <td>1.399</td>
      <td>-5.683</td>
      <td>-0.850</td>
      <td>-0.022</td>
      <td>0.738</td>
      <td>16.875</td>
    </tr>
    <tr>
      <th>5</th>
      <td>V5</td>
      <td>284315.0</td>
      <td>0.005</td>
      <td>1.357</td>
      <td>-113.743</td>
      <td>-0.689</td>
      <td>-0.053</td>
      <td>0.612</td>
      <td>34.802</td>
    </tr>
    <tr>
      <th>6</th>
      <td>V6</td>
      <td>284315.0</td>
      <td>0.002</td>
      <td>1.330</td>
      <td>-26.161</td>
      <td>-0.767</td>
      <td>-0.273</td>
      <td>0.400</td>
      <td>73.302</td>
    </tr>
    <tr>
      <th>7</th>
      <td>V7</td>
      <td>284315.0</td>
      <td>0.010</td>
      <td>1.179</td>
      <td>-31.765</td>
      <td>-0.551</td>
      <td>0.041</td>
      <td>0.571</td>
      <td>120.589</td>
    </tr>
    <tr>
      <th>8</th>
      <td>V8</td>
      <td>284315.0</td>
      <td>-0.001</td>
      <td>1.161</td>
      <td>-73.217</td>
      <td>-0.209</td>
      <td>0.022</td>
      <td>0.326</td>
      <td>18.709</td>
    </tr>
    <tr>
      <th>9</th>
      <td>V9</td>
      <td>284315.0</td>
      <td>0.004</td>
      <td>1.089</td>
      <td>-6.291</td>
      <td>-0.640</td>
      <td>-0.050</td>
      <td>0.598</td>
      <td>15.595</td>
    </tr>
    <tr>
      <th>10</th>
      <td>V10</td>
      <td>284315.0</td>
      <td>0.010</td>
      <td>1.044</td>
      <td>-14.741</td>
      <td>-0.533</td>
      <td>-0.092</td>
      <td>0.455</td>
      <td>23.745</td>
    </tr>
    <tr>
      <th>11</th>
      <td>V11</td>
      <td>284315.0</td>
      <td>-0.007</td>
      <td>1.003</td>
      <td>-4.797</td>
      <td>-0.763</td>
      <td>-0.035</td>
      <td>0.736</td>
      <td>10.002</td>
    </tr>
    <tr>
      <th>12</th>
      <td>V12</td>
      <td>284315.0</td>
      <td>0.011</td>
      <td>0.946</td>
      <td>-15.145</td>
      <td>-0.402</td>
      <td>0.142</td>
      <td>0.619</td>
      <td>7.848</td>
    </tr>
    <tr>
      <th>13</th>
      <td>V13</td>
      <td>284315.0</td>
      <td>0.000</td>
      <td>0.995</td>
      <td>-5.792</td>
      <td>-0.648</td>
      <td>-0.014</td>
      <td>0.662</td>
      <td>7.127</td>
    </tr>
    <tr>
      <th>14</th>
      <td>V14</td>
      <td>284315.0</td>
      <td>0.012</td>
      <td>0.897</td>
      <td>-18.392</td>
      <td>-0.422</td>
      <td>0.052</td>
      <td>0.494</td>
      <td>10.527</td>
    </tr>
    <tr>
      <th>15</th>
      <td>V15</td>
      <td>284315.0</td>
      <td>0.000</td>
      <td>0.915</td>
      <td>-4.391</td>
      <td>-0.583</td>
      <td>0.048</td>
      <td>0.649</td>
      <td>8.878</td>
    </tr>
    <tr>
      <th>16</th>
      <td>V16</td>
      <td>284315.0</td>
      <td>0.007</td>
      <td>0.845</td>
      <td>-10.116</td>
      <td>-0.466</td>
      <td>0.067</td>
      <td>0.524</td>
      <td>17.315</td>
    </tr>
    <tr>
      <th>17</th>
      <td>V17</td>
      <td>284315.0</td>
      <td>0.012</td>
      <td>0.749</td>
      <td>-17.098</td>
      <td>-0.483</td>
      <td>-0.065</td>
      <td>0.400</td>
      <td>9.254</td>
    </tr>
    <tr>
      <th>18</th>
      <td>V18</td>
      <td>284315.0</td>
      <td>0.004</td>
      <td>0.825</td>
      <td>-5.367</td>
      <td>-0.497</td>
      <td>-0.003</td>
      <td>0.501</td>
      <td>5.041</td>
    </tr>
    <tr>
      <th>19</th>
      <td>V19</td>
      <td>284315.0</td>
      <td>-0.001</td>
      <td>0.812</td>
      <td>-7.214</td>
      <td>-0.456</td>
      <td>0.003</td>
      <td>0.457</td>
      <td>5.592</td>
    </tr>
    <tr>
      <th>20</th>
      <td>V20</td>
      <td>284315.0</td>
      <td>-0.001</td>
      <td>0.769</td>
      <td>-54.498</td>
      <td>-0.212</td>
      <td>-0.063</td>
      <td>0.132</td>
      <td>39.421</td>
    </tr>
    <tr>
      <th>21</th>
      <td>V21</td>
      <td>284315.0</td>
      <td>-0.001</td>
      <td>0.717</td>
      <td>-34.830</td>
      <td>-0.229</td>
      <td>-0.030</td>
      <td>0.186</td>
      <td>22.615</td>
    </tr>
    <tr>
      <th>22</th>
      <td>V22</td>
      <td>284315.0</td>
      <td>-0.000</td>
      <td>0.724</td>
      <td>-10.933</td>
      <td>-0.542</td>
      <td>0.007</td>
      <td>0.528</td>
      <td>10.503</td>
    </tr>
    <tr>
      <th>23</th>
      <td>V23</td>
      <td>284315.0</td>
      <td>0.000</td>
      <td>0.622</td>
      <td>-44.808</td>
      <td>-0.162</td>
      <td>-0.011</td>
      <td>0.148</td>
      <td>22.528</td>
    </tr>
    <tr>
      <th>24</th>
      <td>V24</td>
      <td>284315.0</td>
      <td>0.000</td>
      <td>0.606</td>
      <td>-2.837</td>
      <td>-0.354</td>
      <td>0.041</td>
      <td>0.440</td>
      <td>4.585</td>
    </tr>
    <tr>
      <th>25</th>
      <td>V25</td>
      <td>284315.0</td>
      <td>-0.000</td>
      <td>0.521</td>
      <td>-10.295</td>
      <td>-0.317</td>
      <td>0.016</td>
      <td>0.351</td>
      <td>7.520</td>
    </tr>
    <tr>
      <th>26</th>
      <td>V26</td>
      <td>284315.0</td>
      <td>-0.000</td>
      <td>0.482</td>
      <td>-2.605</td>
      <td>-0.327</td>
      <td>-0.052</td>
      <td>0.241</td>
      <td>3.517</td>
    </tr>
    <tr>
      <th>27</th>
      <td>V27</td>
      <td>284315.0</td>
      <td>-0.000</td>
      <td>0.400</td>
      <td>-22.566</td>
      <td>-0.071</td>
      <td>0.001</td>
      <td>0.091</td>
      <td>31.612</td>
    </tr>
    <tr>
      <th>28</th>
      <td>V28</td>
      <td>284315.0</td>
      <td>-0.000</td>
      <td>0.330</td>
      <td>-15.430</td>
      <td>-0.053</td>
      <td>0.011</td>
      <td>0.078</td>
      <td>33.848</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Amount</td>
      <td>284315.0</td>
      <td>88.291</td>
      <td>250.105</td>
      <td>0.000</td>
      <td>5.650</td>
      <td>22.000</td>
      <td>77.050</td>
      <td>25691.160</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Class</td>
      <td>284315.0</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-5483646c-831b-4623-a15b-2fcc096086e4')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-5483646c-831b-4623-a15b-2fcc096086e4 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-5483646c-831b-4623-a15b-2fcc096086e4');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code execution_count="24" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="0QP7Z2HsUW43" outputId="b756c6ae-2853-4ec5-89e8-ec3ae88ef11c"}
``` {.python}
# Percentages of fraudulent and non-fradulent transactions in data
print(f'Percent of Non-Fraudulent Transactions = {round(df["Class"].value_counts()[0]/len(df) * 100,3)}%')
print(f'Percent of Fraudulent Transactions = {round(df["Class"].value_counts()[1]/len(df) * 100,3)}%')
```

::: {.output .stream .stdout}
    Percent of Non-Fraudulent Transactions = 99.827%
    Percent of Fraudulent Transactions = 0.173%
:::
:::

::: {.cell .code execution_count="25" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="3gVXGJ8oWjEu" outputId="f82f1c08-bdd1-4d5a-c81c-0ef2bf2901e9"}
``` {.python}
print('\n\033[1m  Fraudulent Transaction Distribution by amount \033[0m')
print("-"*50)
print(df[(df['Class'] == 1)]['Amount'].value_counts().head())
```

::: {.output .stream .stdout}

      Fraudulent Transaction Distribution by amount 
    --------------------------------------------------
    1.00     113
    0.00      27
    99.99     27
    0.76      17
    0.77      10
    Name: Amount, dtype: int64
:::
:::

::: {.cell .code execution_count="28" colab="{\"height\":301,\"base_uri\":\"https://localhost:8080/\"}" id="dblfj2iqtGzw" outputId="bd5050f9-199d-4489-d1ee-a2d35250300b"}
``` {.python}
fraud_or_not = df["Class"].value_counts().tolist()
labels = ['Not Fraud','Frauds']
values = [fraud_or_not[0], fraud_or_not[1]]
print('Count of Non Fraud and Fraud are ',values)
colors = ['lightgreen', 'red']
fix, ax1 = plt.subplots()
# the autopct is to get the percentage values.
_, texts, autotexts = ax1.pie(values, labels=labels, colors=colors, startangle = 90,shadow=True, autopct='%.2f')
list(map(lambda x:x.set_fontsize(15), autotexts))
ax1.set_title("Fraud & Not Fraud", fontsize=15)
```

::: {.output .stream .stdout}
    Count of Non Fraud and Fraud are  [284315, 492]
:::

::: {.output .execute_result execution_count="28"}
    Text(0.5, 1.0, 'Fraud & Not Fraud')
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/e8c7cf59488fb4aa6c836cdf3fb569a7f2d70c75.png)
:::
:::

::: {.cell .markdown id="Zka3KUQmS80Z"}
Correlation of Variables with Class
:::

::: {.cell .code execution_count="31" colab="{\"height\":884,\"base_uri\":\"https://localhost:8080/\"}" id="tBIFlPoQW4Md" outputId="da5cc569-2827-4d7f-8582-dea2dfc6e711"}
``` {.python}
corr = df.corrwith(df['Class']).reset_index()
corr.columns = ['Index','Correlations']
corr = corr.set_index('Index')
corr = corr.sort_values(by=['Correlations'], ascending = False)
plt.figure(figsize=(8, 15))
fig = sns.heatmap(corr, annot=True, fmt="g", linewidths=0.4)
plt.title("Correlation of Variables with Class", fontsize=20)
plt.show()
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/b87aae39c242c09fd0f8b42424b06da9edd88930.png)
:::
:::

::: {.cell .markdown id="4SZReShcT4iZ"}
Correlation of Variables with each other
:::

::: {.cell .code execution_count="32" colab="{\"height\":660,\"base_uri\":\"https://localhost:8080/\"}" id="rteNIpnPkuYG" outputId="3ffa1760-cfaa-4fa7-c810-1fb761ca4767"}
``` {.python}
plt.figure(figsize=(15,10))
plt.title("Correlation of Variables with each other", fontsize=20 )
sns.heatmap(df.corr(), annot = True, fmt = '.1f' )
```

::: {.output .execute_result execution_count="32"}
    <matplotlib.axes._subplots.AxesSubplot at 0x7f1444dc2590>
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/75abff461a5018b34f80a9529bfcd76d706354a3.png)
:::
:::

::: {.cell .code execution_count="33" id="k4FV7if2PXaY"}
``` {.python}
# shuffles the data before undersample
# to understand better the relation between variables 
df = df.sample(frac=1)
# Takes 492 of non-fraud rows
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]
# Creating a new dataframe with 50:50 ratio of fraud:non-fraud
normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
# Shufffle the new dataframe
new_df = normal_distributed_df.sample(frac=1, random_state=42)
```
:::

::: {.cell .code execution_count="34" colab="{\"height\":660,\"base_uri\":\"https://localhost:8080/\"}" id="TUwQh4x-G8fl" outputId="88f7b24b-3929-47be-9ecb-45e1a42e4f24"}
``` {.python}
plt.figure(figsize=(15,10))
plt.title("Correlation of Variables with each other", fontsize=20 )
sns.heatmap(new_df.corr(), annot = True, fmt = '.1f' )
```

::: {.output .execute_result execution_count="34"}
    <matplotlib.axes._subplots.AxesSubplot at 0x7f1444e1ab10>
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/92e4f835da2e2ced1afc892aa32afe4366151435.png)
:::
:::

::: {.cell .code execution_count="35" colab="{\"height\":884,\"base_uri\":\"https://localhost:8080/\"}" id="sphlTxJOUm1g" outputId="c3357417-d059-4ff5-86bd-38e28fba120c"}
``` {.python}
corr = new_df.corrwith(df['Class']).reset_index()
corr.columns = ['Index','Correlations']
corr = corr.set_index('Index')
corr = corr.sort_values(by=['Correlations'], ascending = False)
plt.figure(figsize=(8, 15))
fig = sns.heatmap(corr, annot=True, fmt="g", linewidths=0.4)
plt.title("Correlation of Variables with Class", fontsize=20)
plt.show()
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/1bbbbb1216ddbccdc7c6ef68af206c439b50be99.png)
:::
:::

::: {.cell .code execution_count="38" colab="{\"height\":295,\"base_uri\":\"https://localhost:8080/\"}" id="8SGZjmBmM7Eg" outputId="1f6345e5-9103-4bb0-acd2-2ee33a5b219d"}
``` {.python}
''''
We can see that the features selected here have a statistically higher value when there is a fraudulent transaction. We don't know what these features actually mean, but we (at least) 
have somewhat of an understanding of their correlation with the class. 
This can be very useful when we are further preparing our dataset for our model.
'''
f, axes = plt.subplots(ncols=3, figsize=(20,4))

# Creating the boxplot
sns.boxplot(x="Class", y="V2", data=new_df, palette=colors, ax=axes[0])
axes[0].set_title('V2 vs Class (Positive Correlation)')

sns.boxplot(x="Class", y="V4", data=new_df, palette=colors, ax=axes[1])
axes[1].set_title('V4 vs Class (Positive Correlation)')

sns.boxplot(x="Class", y="V11", data=new_df, palette=colors, ax=axes[2])
axes[2].set_title('V11 vs Class (Positive Correlation)')



plt.show()
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/cc907e3386ea377591064b720c7a523eeee8c0b8.png)
:::
:::

::: {.cell .code execution_count="41" colab="{\"height\":295,\"base_uri\":\"https://localhost:8080/\"}" id="7dexLy46NuPZ" outputId="bae897a7-0c12-4112-e9a6-26cc956612be"}
``` {.python}
f, axes = plt.subplots(ncols=3, figsize=(20,4))
# Creating the boxplot (negative correlation)
sns.boxplot(x="Class", y="V10", data=new_df, palette=colors, ax=axes[0])
axes[0].set_title('V10 vs Class (Negative Correlation)')

sns.boxplot(x="Class", y="V12", data=new_df, palette=colors, ax=axes[1])
axes[1].set_title('V12 vs Class (Negative Correlation)')

sns.boxplot(x="Class", y="V14", data=new_df, palette=colors, ax=axes[2])
axes[2].set_title('V14 vs Class (Negative Correlation)')

plt.show()
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/6233e147c925a69f18263444db47400b7858127b.png)
:::
:::

::: {.cell .markdown id="rmKl-6wxUs5G"}
T-SNE
:::

::: {.cell .code execution_count="42" id="YOgBk4heOdCv"}
``` {.python}
# let's update our inputs and outputs
standard_scaler = StandardScaler()
X = new_df.drop('Class', axis=1)
y = new_df['Class']
#Scale features to improve the training ability of TSNE.
X_Scaled = standard_scaler.fit_transform(X.values)

# t-SNE
X_reduced_tsne = TSNE(n_components=2, random_state=2).fit_transform(X_Scaled)
```
:::

::: {.cell .code execution_count="43" colab="{\"height\":295,\"base_uri\":\"https://localhost:8080/\"}" id="VgGM8dVFQixi" outputId="c415c1f9-c67d-41ed-faeb-e8ceb2822b39"}
``` {.python}
#Most of the fraudulent transactions are well separated in the original dataset sample in this t-SNE plot, 
#while some are mixed within the rest of the data.
color_map = {1:'red', 0:'blue'}
plt.figure()
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X_reduced_tsne[y==cl,0], 
                y=X_reduced_tsne[y==cl,1], 
                c=color_map[idx], 
                label=cl)
plt.xlabel('X ')
plt.ylabel('Y ')
plt.legend(loc='best')
plt.title('t-SNE visualization of Undersampled data')
plt.show()
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/0a9df6459b2256659a5027645eac15b5eb6d971c.png)
:::
:::

::: {.cell .markdown id="AtJgQTVhU6bM"}
Comparing Perameters with Number of Transactions
:::

::: {.cell .code execution_count="44" colab="{\"height\":618,\"base_uri\":\"https://localhost:8080/\"}" id="8rVWObYcXi-Q" outputId="b5aa357f-319f-47ca-eda0-0d21ad31674c"}
``` {.python}
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20,10))

fraud_transactions = df.Time[df.Class == 1]
normal_transactions = df.Time[df.Class == 0]

ax1.hist(fraud_transactions, bins = 50, edgecolor="black")
ax1.set_xlim([min(fraud_transactions), max(fraud_transactions)])
ax1.set_title('Fraudulent Transactions', fontsize=15)
ax1.set_ylabel("Number of Transactions",  fontsize=13)

ax2.hist(normal_transactions, bins = 50, color='deepskyblue', edgecolor="black")
ax2.set_xlim([min(normal_transactions), max(normal_transactions)])
ax2.set_title('Normal Transactions',  fontsize=15)

ax2.set_xlabel('Time (in Seconds)',  fontsize=13)
ax2.set_ylabel('Number of Transactions',  fontsize=13)

plt.show()
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/d50f1f032c25ae89dc77c49ef49982d22e3c779c.png)
:::
:::

::: {.cell .code execution_count="46" id="t8squj4zjFGS"}
``` {.python}
# converting seconds to time delta to extract hours and mins

timedelta = pd.to_timedelta(df['Time'], unit='s')

df['mins'] = (timedelta.dt.components.minutes).astype(int)
df['hours'] = (timedelta.dt.components.hours).astype(int)
```
:::

::: {.cell .code execution_count="47" colab="{\"height\":1000,\"base_uri\":\"https://localhost:8080/\"}" id="EbR5R2VejPTA" outputId="4aa7041f-eadc-49b1-cd67-232cf7b85340"}
``` {.python}
# Countplots for hours vs count of transactions

fig, axs = plt.subplots(3, figsize=(12,20))

fig.subplots_adjust(hspace=.5)

sns.countplot(df['hours'], ax = axs[0], palette='Set3')
axs[0].set_title("Distribution of Total Transactions",fontsize=20)
axs[0].set_facecolor("white")

sns.countplot(df[(df['Class'] == 1)]['hours'], ax=axs[1],  palette='Set3')
axs[1].set_title("Distribution of Fraudulent Transactions", fontsize=20)
axs[1].set_facecolor('white')

sns.countplot(df[(df['Class'] == 0)]['hours'], ax=axs[2], palette='Set3')
axs[2].set_title("Distribution of Normal Transactions", fontsize=20)
axs[2].set_facecolor("white")

plt.show()
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/e181f01ee81ec80ebb1857821309c64c415aed02.png)
:::
:::

::: {.cell .code execution_count="49" colab="{\"height\":515,\"base_uri\":\"https://localhost:8080/\"}" id="Dq61KdXGkI8L" outputId="9b160528-422e-4268-b836-bcf4c054655c"}
``` {.python}
#Class and Amount vs Time
# Scatter plot of Class vs Amount and Time for Normal Transactions 

plt.figure(figsize=(20,8))
fig = plt.scatter(x=df[df['Class'] == 0]['Time'], y=df[df['Class'] == 0]['Amount'], color="yellow", s=50, edgecolor='red')
plt.title("Time vs Transaction Amount in Normal Transactions", fontsize=20)
plt.xlabel("Time (in seconds)", fontsize=13)
plt.ylabel("Amount of Transaction", fontsize=13)

plt.show()
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/d6d77920428b18e60da497cd2b1df104803c5b7a.png)
:::
:::

::: {.cell .code execution_count="52" colab="{\"height\":517,\"base_uri\":\"https://localhost:8080/\"}" id="o7CH6qq7kcRY" outputId="0412b5c0-4998-46fa-a8db-df7ddeca0d98"}
``` {.python}
# Scatter plot of Class vs Amount and Time for Fraudulent Transactions 

plt.figure(figsize=(20,8))
fig = plt.scatter(x=df[df['Class'] == 1]['Time'], y=df[df['Class'] == 1]['Amount'], color="r", s=80)
plt.title("Time vs Transaction Amount in Fraud Cases", fontsize=20)
plt.xlabel("Time (in seconds)", fontsize=13)
plt.ylabel("Amount of Transaction", fontsize=13)

plt.show()
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/f3b6724d99aace1a7d2ff2806d5a40d99c8d5c8c.png)
:::
:::

::: {.cell .markdown id="uebe240PXjd8"}
Looking the V\'s features
:::

::: {.cell .code execution_count="55" colab="{\"height\":1000,\"base_uri\":\"https://localhost:8080/\"}" id="scIggbUrmfEB" outputId="71d99468-fd6f-4772-86e8-f3ce100417cf"}
``` {.python}
columns = df.iloc[:,1:29].columns

frauds = df.Class == 1
normals = df.Class == 0

grid = gridspec.GridSpec(14, 2)
plt.figure(figsize=(20,20*4))

for n, col in enumerate(df[columns]):
    ax = plt.subplot(grid[n])
    sns.distplot(df[col][frauds], color='red', kde_kws={"color": "k", "lw": 1.5},  hist_kws=dict(alpha=1)) 
    sns.distplot(df[col][normals],color='lightblue', kde_kws={"color": "k", "lw": 1.5},  hist_kws=dict(alpha=1))
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title(str(col), fontsize=20)
    ax.set_xlabel('')
plt.show()
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/17dc8903e83caa3dfd80b4d23cf0a423f7d4fefc.png)
:::
:::

::: {.cell .code execution_count="56" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="UuauPASYncKB" outputId="32b0baff-bfc0-4280-b557-be927af211af"}
``` {.python}
#Data Cleaning and Preprocessing
# Finding the 3rd and 1st Quantile for Amount Column
Q3 = np.percentile(df['Amount'], 75)
Q1 = np.percentile(df['Amount'], 25)

# setting the cutoff
cutoff = 7.0
# computing the interquartile range
IQR = (Q3 - Q1)

# computing lower bound and upper bound
lower_bound = Q1 - (IQR * cutoff)
upper_bound = Q3 + (IQR * cutoff)

# creating a filter to remove values less than lower bound and greater than
# upper bound
filter_data = (df['Amount'] < lower_bound) | (df['Amount'] > upper_bound)

# filtering data
outliers = df[filter_data]['Amount']
fraud_outliers = df[(df['Class'] == 1) & filter_data]['Amount']
normal_outliers = df[(df['Class'] == 0) & filter_data]['Amount']

print(f"Total Number of Outliers : {outliers.count()}")
print(f"Number of Outliers in Fraudulent Class : {fraud_outliers.count()}")
print(f"No of Outliers in Normal Class : {normal_outliers.count()}")
print(f"Percentage of Fraud amount outliers : {round((fraud_outliers.count()/outliers.count())*100,2)}%")
```

::: {.output .stream .stdout}
    Total Number of Outliers : 7478
    Number of Outliers in Fraudulent Class : 29
    No of Outliers in Normal Class : 7449
    Percentage of Fraud amount outliers : 0.39%
:::
:::

::: {.cell .code execution_count="57" id="S_PM4Sb9n5ZF"}
``` {.python}
data = df.drop(outliers.index)
data.reset_index(inplace=True, drop=True)
```
:::

::: {.cell .code execution_count="58" colab="{\"height\":487,\"base_uri\":\"https://localhost:8080/\"}" id="Dbp73W5WmjHg" outputId="0d3ced5e-d7eb-4e7f-be1c-2c2b433518cb"}
``` {.python}
data
```

::: {.output .execute_result execution_count="58"}
```{=html}
  <div id="df-8cf74141-47a3-445a-abb0-4e13c8fb648f">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
      <th>mins</th>
      <th>hours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>115960.0</td>
      <td>0.122040</td>
      <td>0.889577</td>
      <td>-0.557672</td>
      <td>-0.739899</td>
      <td>1.076581</td>
      <td>-0.251537</td>
      <td>0.768230</td>
      <td>0.150525</td>
      <td>-0.109897</td>
      <td>...</td>
      <td>0.055476</td>
      <td>0.085934</td>
      <td>-0.457265</td>
      <td>0.126185</td>
      <td>0.214236</td>
      <td>0.067386</td>
      <td>1.79</td>
      <td>0</td>
      <td>12</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>138396.0</td>
      <td>1.983664</td>
      <td>-0.144566</td>
      <td>-1.601533</td>
      <td>0.650530</td>
      <td>-0.058817</td>
      <td>-0.920016</td>
      <td>-0.079494</td>
      <td>-0.052650</td>
      <td>0.791831</td>
      <td>...</td>
      <td>0.007543</td>
      <td>-0.045930</td>
      <td>0.086611</td>
      <td>0.712647</td>
      <td>-0.066077</td>
      <td>-0.058296</td>
      <td>5.74</td>
      <td>0</td>
      <td>26</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>58346.0</td>
      <td>-3.095610</td>
      <td>2.341266</td>
      <td>0.978825</td>
      <td>1.302911</td>
      <td>-1.550850</td>
      <td>0.900366</td>
      <td>-1.023949</td>
      <td>1.546270</td>
      <td>1.244595</td>
      <td>...</td>
      <td>0.074944</td>
      <td>0.194447</td>
      <td>0.033323</td>
      <td>-0.509168</td>
      <td>-0.694157</td>
      <td>-0.489850</td>
      <td>5.67</td>
      <td>0</td>
      <td>12</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>78116.0</td>
      <td>-0.677805</td>
      <td>-0.323111</td>
      <td>1.795775</td>
      <td>-2.280477</td>
      <td>-0.424261</td>
      <td>-1.015679</td>
      <td>0.842032</td>
      <td>-0.386281</td>
      <td>-1.098708</td>
      <td>...</td>
      <td>0.226603</td>
      <td>0.365986</td>
      <td>-0.261502</td>
      <td>-0.817321</td>
      <td>-0.108201</td>
      <td>-0.071991</td>
      <td>126.00</td>
      <td>0</td>
      <td>41</td>
      <td>21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11054.0</td>
      <td>1.365524</td>
      <td>-0.193916</td>
      <td>0.327665</td>
      <td>-0.513938</td>
      <td>-0.639434</td>
      <td>-0.543843</td>
      <td>-0.733840</td>
      <td>-0.151407</td>
      <td>0.173629</td>
      <td>...</td>
      <td>-0.056541</td>
      <td>-0.104268</td>
      <td>0.386347</td>
      <td>-0.321959</td>
      <td>0.002022</td>
      <td>0.025594</td>
      <td>15.95</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>277324</th>
      <td>67274.0</td>
      <td>1.145484</td>
      <td>-0.781774</td>
      <td>0.488417</td>
      <td>-0.859133</td>
      <td>-0.543703</td>
      <td>1.031799</td>
      <td>-0.869260</td>
      <td>0.403027</td>
      <td>2.153869</td>
      <td>...</td>
      <td>-0.340819</td>
      <td>-1.081764</td>
      <td>0.825123</td>
      <td>-0.393337</td>
      <td>0.113609</td>
      <td>0.009240</td>
      <td>37.00</td>
      <td>0</td>
      <td>41</td>
      <td>18</td>
    </tr>
    <tr>
      <th>277325</th>
      <td>14463.0</td>
      <td>1.205584</td>
      <td>-0.018865</td>
      <td>0.851469</td>
      <td>0.271141</td>
      <td>-0.611746</td>
      <td>-0.355526</td>
      <td>-0.492331</td>
      <td>-0.082980</td>
      <td>1.680204</td>
      <td>...</td>
      <td>0.011242</td>
      <td>0.113186</td>
      <td>0.146389</td>
      <td>1.059321</td>
      <td>-0.070533</td>
      <td>0.003090</td>
      <td>14.95</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>277326</th>
      <td>138850.0</td>
      <td>1.823779</td>
      <td>-0.790119</td>
      <td>-0.079887</td>
      <td>0.486926</td>
      <td>-0.992713</td>
      <td>0.013298</td>
      <td>-0.943318</td>
      <td>0.249591</td>
      <td>1.273188</td>
      <td>...</td>
      <td>0.226039</td>
      <td>-0.361329</td>
      <td>-0.592829</td>
      <td>0.166075</td>
      <td>-0.015923</td>
      <td>-0.039252</td>
      <td>69.88</td>
      <td>0</td>
      <td>34</td>
      <td>14</td>
    </tr>
    <tr>
      <th>277327</th>
      <td>48643.0</td>
      <td>-3.729045</td>
      <td>-3.345294</td>
      <td>0.504541</td>
      <td>1.349409</td>
      <td>2.587102</td>
      <td>-2.875825</td>
      <td>-1.848069</td>
      <td>0.775216</td>
      <td>0.299077</td>
      <td>...</td>
      <td>0.036287</td>
      <td>0.551506</td>
      <td>-0.742764</td>
      <td>0.095890</td>
      <td>0.128720</td>
      <td>-0.504839</td>
      <td>12.31</td>
      <td>0</td>
      <td>30</td>
      <td>13</td>
    </tr>
    <tr>
      <th>277328</th>
      <td>161884.0</td>
      <td>0.245442</td>
      <td>0.757524</td>
      <td>0.069692</td>
      <td>-0.675437</td>
      <td>0.834176</td>
      <td>-0.494112</td>
      <td>1.009999</td>
      <td>-0.188276</td>
      <td>-0.293707</td>
      <td>...</td>
      <td>0.116291</td>
      <td>0.624563</td>
      <td>-0.456012</td>
      <td>0.097241</td>
      <td>-0.027043</td>
      <td>0.000804</td>
      <td>12.27</td>
      <td>0</td>
      <td>58</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
<p>277329 rows × 33 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-8cf74141-47a3-445a-abb0-4e13c8fb648f')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-8cf74141-47a3-445a-abb0-4e13c8fb648f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-8cf74141-47a3-445a-abb0-4e13c8fb648f');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code execution_count="59" colab="{\"height\":226,\"base_uri\":\"https://localhost:8080/\"}" id="a-EO-VQ8n9kq" outputId="bf140900-9034-4206-dd65-0ff3777330e0"}
``` {.python}
data.head().style.set_properties(**{"background-color":"white",
                           "color" : "black"})
```

::: {.output .execute_result execution_count="59"}
```{=html}
<style type="text/css">
#T_a6ff1_row0_col0, #T_a6ff1_row0_col1, #T_a6ff1_row0_col2, #T_a6ff1_row0_col3, #T_a6ff1_row0_col4, #T_a6ff1_row0_col5, #T_a6ff1_row0_col6, #T_a6ff1_row0_col7, #T_a6ff1_row0_col8, #T_a6ff1_row0_col9, #T_a6ff1_row0_col10, #T_a6ff1_row0_col11, #T_a6ff1_row0_col12, #T_a6ff1_row0_col13, #T_a6ff1_row0_col14, #T_a6ff1_row0_col15, #T_a6ff1_row0_col16, #T_a6ff1_row0_col17, #T_a6ff1_row0_col18, #T_a6ff1_row0_col19, #T_a6ff1_row0_col20, #T_a6ff1_row0_col21, #T_a6ff1_row0_col22, #T_a6ff1_row0_col23, #T_a6ff1_row0_col24, #T_a6ff1_row0_col25, #T_a6ff1_row0_col26, #T_a6ff1_row0_col27, #T_a6ff1_row0_col28, #T_a6ff1_row0_col29, #T_a6ff1_row0_col30, #T_a6ff1_row0_col31, #T_a6ff1_row0_col32, #T_a6ff1_row1_col0, #T_a6ff1_row1_col1, #T_a6ff1_row1_col2, #T_a6ff1_row1_col3, #T_a6ff1_row1_col4, #T_a6ff1_row1_col5, #T_a6ff1_row1_col6, #T_a6ff1_row1_col7, #T_a6ff1_row1_col8, #T_a6ff1_row1_col9, #T_a6ff1_row1_col10, #T_a6ff1_row1_col11, #T_a6ff1_row1_col12, #T_a6ff1_row1_col13, #T_a6ff1_row1_col14, #T_a6ff1_row1_col15, #T_a6ff1_row1_col16, #T_a6ff1_row1_col17, #T_a6ff1_row1_col18, #T_a6ff1_row1_col19, #T_a6ff1_row1_col20, #T_a6ff1_row1_col21, #T_a6ff1_row1_col22, #T_a6ff1_row1_col23, #T_a6ff1_row1_col24, #T_a6ff1_row1_col25, #T_a6ff1_row1_col26, #T_a6ff1_row1_col27, #T_a6ff1_row1_col28, #T_a6ff1_row1_col29, #T_a6ff1_row1_col30, #T_a6ff1_row1_col31, #T_a6ff1_row1_col32, #T_a6ff1_row2_col0, #T_a6ff1_row2_col1, #T_a6ff1_row2_col2, #T_a6ff1_row2_col3, #T_a6ff1_row2_col4, #T_a6ff1_row2_col5, #T_a6ff1_row2_col6, #T_a6ff1_row2_col7, #T_a6ff1_row2_col8, #T_a6ff1_row2_col9, #T_a6ff1_row2_col10, #T_a6ff1_row2_col11, #T_a6ff1_row2_col12, #T_a6ff1_row2_col13, #T_a6ff1_row2_col14, #T_a6ff1_row2_col15, #T_a6ff1_row2_col16, #T_a6ff1_row2_col17, #T_a6ff1_row2_col18, #T_a6ff1_row2_col19, #T_a6ff1_row2_col20, #T_a6ff1_row2_col21, #T_a6ff1_row2_col22, #T_a6ff1_row2_col23, #T_a6ff1_row2_col24, #T_a6ff1_row2_col25, #T_a6ff1_row2_col26, #T_a6ff1_row2_col27, #T_a6ff1_row2_col28, #T_a6ff1_row2_col29, #T_a6ff1_row2_col30, #T_a6ff1_row2_col31, #T_a6ff1_row2_col32, #T_a6ff1_row3_col0, #T_a6ff1_row3_col1, #T_a6ff1_row3_col2, #T_a6ff1_row3_col3, #T_a6ff1_row3_col4, #T_a6ff1_row3_col5, #T_a6ff1_row3_col6, #T_a6ff1_row3_col7, #T_a6ff1_row3_col8, #T_a6ff1_row3_col9, #T_a6ff1_row3_col10, #T_a6ff1_row3_col11, #T_a6ff1_row3_col12, #T_a6ff1_row3_col13, #T_a6ff1_row3_col14, #T_a6ff1_row3_col15, #T_a6ff1_row3_col16, #T_a6ff1_row3_col17, #T_a6ff1_row3_col18, #T_a6ff1_row3_col19, #T_a6ff1_row3_col20, #T_a6ff1_row3_col21, #T_a6ff1_row3_col22, #T_a6ff1_row3_col23, #T_a6ff1_row3_col24, #T_a6ff1_row3_col25, #T_a6ff1_row3_col26, #T_a6ff1_row3_col27, #T_a6ff1_row3_col28, #T_a6ff1_row3_col29, #T_a6ff1_row3_col30, #T_a6ff1_row3_col31, #T_a6ff1_row3_col32, #T_a6ff1_row4_col0, #T_a6ff1_row4_col1, #T_a6ff1_row4_col2, #T_a6ff1_row4_col3, #T_a6ff1_row4_col4, #T_a6ff1_row4_col5, #T_a6ff1_row4_col6, #T_a6ff1_row4_col7, #T_a6ff1_row4_col8, #T_a6ff1_row4_col9, #T_a6ff1_row4_col10, #T_a6ff1_row4_col11, #T_a6ff1_row4_col12, #T_a6ff1_row4_col13, #T_a6ff1_row4_col14, #T_a6ff1_row4_col15, #T_a6ff1_row4_col16, #T_a6ff1_row4_col17, #T_a6ff1_row4_col18, #T_a6ff1_row4_col19, #T_a6ff1_row4_col20, #T_a6ff1_row4_col21, #T_a6ff1_row4_col22, #T_a6ff1_row4_col23, #T_a6ff1_row4_col24, #T_a6ff1_row4_col25, #T_a6ff1_row4_col26, #T_a6ff1_row4_col27, #T_a6ff1_row4_col28, #T_a6ff1_row4_col29, #T_a6ff1_row4_col30, #T_a6ff1_row4_col31, #T_a6ff1_row4_col32 {
  background-color: white;
  color: black;
}
</style>
<table id="T_a6ff1_" class="dataframe">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th class="col_heading level0 col0" >Time</th>
      <th class="col_heading level0 col1" >V1</th>
      <th class="col_heading level0 col2" >V2</th>
      <th class="col_heading level0 col3" >V3</th>
      <th class="col_heading level0 col4" >V4</th>
      <th class="col_heading level0 col5" >V5</th>
      <th class="col_heading level0 col6" >V6</th>
      <th class="col_heading level0 col7" >V7</th>
      <th class="col_heading level0 col8" >V8</th>
      <th class="col_heading level0 col9" >V9</th>
      <th class="col_heading level0 col10" >V10</th>
      <th class="col_heading level0 col11" >V11</th>
      <th class="col_heading level0 col12" >V12</th>
      <th class="col_heading level0 col13" >V13</th>
      <th class="col_heading level0 col14" >V14</th>
      <th class="col_heading level0 col15" >V15</th>
      <th class="col_heading level0 col16" >V16</th>
      <th class="col_heading level0 col17" >V17</th>
      <th class="col_heading level0 col18" >V18</th>
      <th class="col_heading level0 col19" >V19</th>
      <th class="col_heading level0 col20" >V20</th>
      <th class="col_heading level0 col21" >V21</th>
      <th class="col_heading level0 col22" >V22</th>
      <th class="col_heading level0 col23" >V23</th>
      <th class="col_heading level0 col24" >V24</th>
      <th class="col_heading level0 col25" >V25</th>
      <th class="col_heading level0 col26" >V26</th>
      <th class="col_heading level0 col27" >V27</th>
      <th class="col_heading level0 col28" >V28</th>
      <th class="col_heading level0 col29" >Amount</th>
      <th class="col_heading level0 col30" >Class</th>
      <th class="col_heading level0 col31" >mins</th>
      <th class="col_heading level0 col32" >hours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_a6ff1_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_a6ff1_row0_col0" class="data row0 col0" >115960.000000</td>
      <td id="T_a6ff1_row0_col1" class="data row0 col1" >0.122040</td>
      <td id="T_a6ff1_row0_col2" class="data row0 col2" >0.889577</td>
      <td id="T_a6ff1_row0_col3" class="data row0 col3" >-0.557672</td>
      <td id="T_a6ff1_row0_col4" class="data row0 col4" >-0.739899</td>
      <td id="T_a6ff1_row0_col5" class="data row0 col5" >1.076581</td>
      <td id="T_a6ff1_row0_col6" class="data row0 col6" >-0.251537</td>
      <td id="T_a6ff1_row0_col7" class="data row0 col7" >0.768230</td>
      <td id="T_a6ff1_row0_col8" class="data row0 col8" >0.150525</td>
      <td id="T_a6ff1_row0_col9" class="data row0 col9" >-0.109897</td>
      <td id="T_a6ff1_row0_col10" class="data row0 col10" >-0.638497</td>
      <td id="T_a6ff1_row0_col11" class="data row0 col11" >0.539103</td>
      <td id="T_a6ff1_row0_col12" class="data row0 col12" >-0.138719</td>
      <td id="T_a6ff1_row0_col13" class="data row0 col13" >-1.166615</td>
      <td id="T_a6ff1_row0_col14" class="data row0 col14" >-0.608435</td>
      <td id="T_a6ff1_row0_col15" class="data row0 col15" >-0.740099</td>
      <td id="T_a6ff1_row0_col16" class="data row0 col16" >0.628549</td>
      <td id="T_a6ff1_row0_col17" class="data row0 col17" >0.105929</td>
      <td id="T_a6ff1_row0_col18" class="data row0 col18" >0.448800</td>
      <td id="T_a6ff1_row0_col19" class="data row0 col19" >0.133305</td>
      <td id="T_a6ff1_row0_col20" class="data row0 col20" >-0.041285</td>
      <td id="T_a6ff1_row0_col21" class="data row0 col21" >-0.321946</td>
      <td id="T_a6ff1_row0_col22" class="data row0 col22" >-0.859182</td>
      <td id="T_a6ff1_row0_col23" class="data row0 col23" >0.055476</td>
      <td id="T_a6ff1_row0_col24" class="data row0 col24" >0.085934</td>
      <td id="T_a6ff1_row0_col25" class="data row0 col25" >-0.457265</td>
      <td id="T_a6ff1_row0_col26" class="data row0 col26" >0.126185</td>
      <td id="T_a6ff1_row0_col27" class="data row0 col27" >0.214236</td>
      <td id="T_a6ff1_row0_col28" class="data row0 col28" >0.067386</td>
      <td id="T_a6ff1_row0_col29" class="data row0 col29" >1.790000</td>
      <td id="T_a6ff1_row0_col30" class="data row0 col30" >0</td>
      <td id="T_a6ff1_row0_col31" class="data row0 col31" >12</td>
      <td id="T_a6ff1_row0_col32" class="data row0 col32" >8</td>
    </tr>
    <tr>
      <th id="T_a6ff1_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_a6ff1_row1_col0" class="data row1 col0" >138396.000000</td>
      <td id="T_a6ff1_row1_col1" class="data row1 col1" >1.983664</td>
      <td id="T_a6ff1_row1_col2" class="data row1 col2" >-0.144566</td>
      <td id="T_a6ff1_row1_col3" class="data row1 col3" >-1.601533</td>
      <td id="T_a6ff1_row1_col4" class="data row1 col4" >0.650530</td>
      <td id="T_a6ff1_row1_col5" class="data row1 col5" >-0.058817</td>
      <td id="T_a6ff1_row1_col6" class="data row1 col6" >-0.920016</td>
      <td id="T_a6ff1_row1_col7" class="data row1 col7" >-0.079494</td>
      <td id="T_a6ff1_row1_col8" class="data row1 col8" >-0.052650</td>
      <td id="T_a6ff1_row1_col9" class="data row1 col9" >0.791831</td>
      <td id="T_a6ff1_row1_col10" class="data row1 col10" >-0.250719</td>
      <td id="T_a6ff1_row1_col11" class="data row1 col11" >0.778779</td>
      <td id="T_a6ff1_row1_col12" class="data row1 col12" >-0.075154</td>
      <td id="T_a6ff1_row1_col13" class="data row1 col13" >-2.119390</td>
      <td id="T_a6ff1_row1_col14" class="data row1 col14" >-0.606401</td>
      <td id="T_a6ff1_row1_col15" class="data row1 col15" >-1.000264</td>
      <td id="T_a6ff1_row1_col16" class="data row1 col16" >0.067119</td>
      <td id="T_a6ff1_row1_col17" class="data row1 col17" >0.856625</td>
      <td id="T_a6ff1_row1_col18" class="data row1 col18" >0.587165</td>
      <td id="T_a6ff1_row1_col19" class="data row1 col19" >0.142988</td>
      <td id="T_a6ff1_row1_col20" class="data row1 col20" >-0.315038</td>
      <td id="T_a6ff1_row1_col21" class="data row1 col21" >0.101805</td>
      <td id="T_a6ff1_row1_col22" class="data row1 col22" >0.502895</td>
      <td id="T_a6ff1_row1_col23" class="data row1 col23" >0.007543</td>
      <td id="T_a6ff1_row1_col24" class="data row1 col24" >-0.045930</td>
      <td id="T_a6ff1_row1_col25" class="data row1 col25" >0.086611</td>
      <td id="T_a6ff1_row1_col26" class="data row1 col26" >0.712647</td>
      <td id="T_a6ff1_row1_col27" class="data row1 col27" >-0.066077</td>
      <td id="T_a6ff1_row1_col28" class="data row1 col28" >-0.058296</td>
      <td id="T_a6ff1_row1_col29" class="data row1 col29" >5.740000</td>
      <td id="T_a6ff1_row1_col30" class="data row1 col30" >0</td>
      <td id="T_a6ff1_row1_col31" class="data row1 col31" >26</td>
      <td id="T_a6ff1_row1_col32" class="data row1 col32" >14</td>
    </tr>
    <tr>
      <th id="T_a6ff1_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_a6ff1_row2_col0" class="data row2 col0" >58346.000000</td>
      <td id="T_a6ff1_row2_col1" class="data row2 col1" >-3.095610</td>
      <td id="T_a6ff1_row2_col2" class="data row2 col2" >2.341266</td>
      <td id="T_a6ff1_row2_col3" class="data row2 col3" >0.978825</td>
      <td id="T_a6ff1_row2_col4" class="data row2 col4" >1.302911</td>
      <td id="T_a6ff1_row2_col5" class="data row2 col5" >-1.550850</td>
      <td id="T_a6ff1_row2_col6" class="data row2 col6" >0.900366</td>
      <td id="T_a6ff1_row2_col7" class="data row2 col7" >-1.023949</td>
      <td id="T_a6ff1_row2_col8" class="data row2 col8" >1.546270</td>
      <td id="T_a6ff1_row2_col9" class="data row2 col9" >1.244595</td>
      <td id="T_a6ff1_row2_col10" class="data row2 col10" >1.157214</td>
      <td id="T_a6ff1_row2_col11" class="data row2 col11" >0.377390</td>
      <td id="T_a6ff1_row2_col12" class="data row2 col12" >1.326861</td>
      <td id="T_a6ff1_row2_col13" class="data row2 col13" >-1.034335</td>
      <td id="T_a6ff1_row2_col14" class="data row2 col14" >-0.217089</td>
      <td id="T_a6ff1_row2_col15" class="data row2 col15" >-1.949684</td>
      <td id="T_a6ff1_row2_col16" class="data row2 col16" >-1.368498</td>
      <td id="T_a6ff1_row2_col17" class="data row2 col17" >1.334146</td>
      <td id="T_a6ff1_row2_col18" class="data row2 col18" >-0.560265</td>
      <td id="T_a6ff1_row2_col19" class="data row2 col19" >1.398303</td>
      <td id="T_a6ff1_row2_col20" class="data row2 col20" >0.309719</td>
      <td id="T_a6ff1_row2_col21" class="data row2 col21" >-0.354588</td>
      <td id="T_a6ff1_row2_col22" class="data row2 col22" >-0.455957</td>
      <td id="T_a6ff1_row2_col23" class="data row2 col23" >0.074944</td>
      <td id="T_a6ff1_row2_col24" class="data row2 col24" >0.194447</td>
      <td id="T_a6ff1_row2_col25" class="data row2 col25" >0.033323</td>
      <td id="T_a6ff1_row2_col26" class="data row2 col26" >-0.509168</td>
      <td id="T_a6ff1_row2_col27" class="data row2 col27" >-0.694157</td>
      <td id="T_a6ff1_row2_col28" class="data row2 col28" >-0.489850</td>
      <td id="T_a6ff1_row2_col29" class="data row2 col29" >5.670000</td>
      <td id="T_a6ff1_row2_col30" class="data row2 col30" >0</td>
      <td id="T_a6ff1_row2_col31" class="data row2 col31" >12</td>
      <td id="T_a6ff1_row2_col32" class="data row2 col32" >16</td>
    </tr>
    <tr>
      <th id="T_a6ff1_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_a6ff1_row3_col0" class="data row3 col0" >78116.000000</td>
      <td id="T_a6ff1_row3_col1" class="data row3 col1" >-0.677805</td>
      <td id="T_a6ff1_row3_col2" class="data row3 col2" >-0.323111</td>
      <td id="T_a6ff1_row3_col3" class="data row3 col3" >1.795775</td>
      <td id="T_a6ff1_row3_col4" class="data row3 col4" >-2.280477</td>
      <td id="T_a6ff1_row3_col5" class="data row3 col5" >-0.424261</td>
      <td id="T_a6ff1_row3_col6" class="data row3 col6" >-1.015679</td>
      <td id="T_a6ff1_row3_col7" class="data row3 col7" >0.842032</td>
      <td id="T_a6ff1_row3_col8" class="data row3 col8" >-0.386281</td>
      <td id="T_a6ff1_row3_col9" class="data row3 col9" >-1.098708</td>
      <td id="T_a6ff1_row3_col10" class="data row3 col10" >-0.234947</td>
      <td id="T_a6ff1_row3_col11" class="data row3 col11" >-0.346110</td>
      <td id="T_a6ff1_row3_col12" class="data row3 col12" >-0.435490</td>
      <td id="T_a6ff1_row3_col13" class="data row3 col13" >0.866670</td>
      <td id="T_a6ff1_row3_col14" class="data row3 col14" >-0.785313</td>
      <td id="T_a6ff1_row3_col15" class="data row3 col15" >-0.436302</td>
      <td id="T_a6ff1_row3_col16" class="data row3 col16" >1.756455</td>
      <td id="T_a6ff1_row3_col17" class="data row3 col17" >-0.642086</td>
      <td id="T_a6ff1_row3_col18" class="data row3 col18" >-1.530292</td>
      <td id="T_a6ff1_row3_col19" class="data row3 col19" >-1.088577</td>
      <td id="T_a6ff1_row3_col20" class="data row3 col20" >0.334034</td>
      <td id="T_a6ff1_row3_col21" class="data row3 col21" >0.141367</td>
      <td id="T_a6ff1_row3_col22" class="data row3 col22" >0.139091</td>
      <td id="T_a6ff1_row3_col23" class="data row3 col23" >0.226603</td>
      <td id="T_a6ff1_row3_col24" class="data row3 col24" >0.365986</td>
      <td id="T_a6ff1_row3_col25" class="data row3 col25" >-0.261502</td>
      <td id="T_a6ff1_row3_col26" class="data row3 col26" >-0.817321</td>
      <td id="T_a6ff1_row3_col27" class="data row3 col27" >-0.108201</td>
      <td id="T_a6ff1_row3_col28" class="data row3 col28" >-0.071991</td>
      <td id="T_a6ff1_row3_col29" class="data row3 col29" >126.000000</td>
      <td id="T_a6ff1_row3_col30" class="data row3 col30" >0</td>
      <td id="T_a6ff1_row3_col31" class="data row3 col31" >41</td>
      <td id="T_a6ff1_row3_col32" class="data row3 col32" >21</td>
    </tr>
    <tr>
      <th id="T_a6ff1_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_a6ff1_row4_col0" class="data row4 col0" >11054.000000</td>
      <td id="T_a6ff1_row4_col1" class="data row4 col1" >1.365524</td>
      <td id="T_a6ff1_row4_col2" class="data row4 col2" >-0.193916</td>
      <td id="T_a6ff1_row4_col3" class="data row4 col3" >0.327665</td>
      <td id="T_a6ff1_row4_col4" class="data row4 col4" >-0.513938</td>
      <td id="T_a6ff1_row4_col5" class="data row4 col5" >-0.639434</td>
      <td id="T_a6ff1_row4_col6" class="data row4 col6" >-0.543843</td>
      <td id="T_a6ff1_row4_col7" class="data row4 col7" >-0.733840</td>
      <td id="T_a6ff1_row4_col8" class="data row4 col8" >-0.151407</td>
      <td id="T_a6ff1_row4_col9" class="data row4 col9" >0.173629</td>
      <td id="T_a6ff1_row4_col10" class="data row4 col10" >-0.002654</td>
      <td id="T_a6ff1_row4_col11" class="data row4 col11" >2.576528</td>
      <td id="T_a6ff1_row4_col12" class="data row4 col12" >-2.135597</td>
      <td id="T_a6ff1_row4_col13" class="data row4 col13" >2.817757</td>
      <td id="T_a6ff1_row4_col14" class="data row4 col14" >0.225021</td>
      <td id="T_a6ff1_row4_col15" class="data row4 col15" >-0.557112</td>
      <td id="T_a6ff1_row4_col16" class="data row4 col16" >1.783841</td>
      <td id="T_a6ff1_row4_col17" class="data row4 col17" >1.218295</td>
      <td id="T_a6ff1_row4_col18" class="data row4 col18" >-0.159246</td>
      <td id="T_a6ff1_row4_col19" class="data row4 col19" >0.534097</td>
      <td id="T_a6ff1_row4_col20" class="data row4 col20" >0.159146</td>
      <td id="T_a6ff1_row4_col21" class="data row4 col21" >-0.050360</td>
      <td id="T_a6ff1_row4_col22" class="data row4 col22" >0.029099</td>
      <td id="T_a6ff1_row4_col23" class="data row4 col23" >-0.056541</td>
      <td id="T_a6ff1_row4_col24" class="data row4 col24" >-0.104268</td>
      <td id="T_a6ff1_row4_col25" class="data row4 col25" >0.386347</td>
      <td id="T_a6ff1_row4_col26" class="data row4 col26" >-0.321959</td>
      <td id="T_a6ff1_row4_col27" class="data row4 col27" >0.002022</td>
      <td id="T_a6ff1_row4_col28" class="data row4 col28" >0.025594</td>
      <td id="T_a6ff1_row4_col29" class="data row4 col29" >15.950000</td>
      <td id="T_a6ff1_row4_col30" class="data row4 col30" >0</td>
      <td id="T_a6ff1_row4_col31" class="data row4 col31" >4</td>
      <td id="T_a6ff1_row4_col32" class="data row4 col32" >3</td>
    </tr>
  </tbody>
</table>
```
:::
:::

::: {.cell .code execution_count="60" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="u-tplgT3oCxh" outputId="3b2153c1-399c-4b05-b4b3-47fa7494932a"}
``` {.python}
data.shape
```

::: {.output .execute_result execution_count="60"}
    (277329, 33)
:::
:::

::: {.cell .code execution_count="61" id="mRbnN_CNoIjs"}
``` {.python}
# applying log transformation of Amount column

data['Amount'] = np.log(data['Amount'] + 0.001)
```
:::

::: {.cell .code execution_count="62" colab="{\"height\":521,\"base_uri\":\"https://localhost:8080/\"}" id="PYYtxZROoL8r" outputId="5a943e72-1662-4ad5-8489-2c3d0c6f92ec"}
``` {.python}
# Box Plot for transformed Amount feature with class

plt.figure(figsize=(12,8))
sns.boxplot(x ="Class", y="Amount", data=data, palette='Set2');
plt.xlabel("Amount", fontsize=13)
plt.ylabel("Class", fontsize=13)
plt.title("Box Plot for Transformed Amount Feature", fontsize=20);
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/51ea72a81be1b0e2f261de8a99e08beac9bf5be8.png)
:::
:::

::: {.cell .code execution_count="64" id="o9ZkMWguhhb4"}
``` {.python}
to_model_cols = data.columns[0:30]
clf = IsolationForest(n_estimators=10, max_samples='auto', contamination=float(.0009), 
                      max_features=1.0, bootstrap=False, n_jobs=-1, random_state=12345, verbose=0)
clf.fit(data[to_model_cols])
pred = clf.predict(data[to_model_cols])
data['Class'] = pred
outliers = data.loc[data['Class']==-1]
outlier_index=list(outliers.index)
```
:::

::: {.cell .markdown id="xwsY660gYjUc"}
Number of anomalies and normal points (points classified as \"-1\" are
anomalous
:::

::: {.cell .code execution_count="65" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="5N5Z2y89Ycw5" outputId="003b635b-644e-46d2-c7fe-194ee307303c"}
``` {.python}
print(data['Class'].value_counts())
```

::: {.output .stream .stdout}
     1    277080
    -1       249
    Name: Class, dtype: int64
:::
:::

::: {.cell .markdown id="WXEIsUADYsOh"}
Scaling the time column
:::

::: {.cell .code execution_count="67" id="JOX5AZikoaW8"}
``` {.python}
robust_scaler = RobustScaler()
data['Time'] = robust_scaler.fit_transform(data['Time'].values.reshape(-1,1))
```
:::

::: {.cell .code colab="{\"height\":487,\"base_uri\":\"https://localhost:8080/\"}" id="GHged-0uoSUr" outputId="d89127a3-3694-4c52-f86a-7fb43adfbd29"}
``` {.python}
data
```

::: {.output .execute_result execution_count="167"}
```{=html}
  <div id="df-27bd2f02-cc2a-41fa-be0d-0c4787c47133">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
      <th>scaled_amount</th>
      <th>scaled_time</th>
      <th>mins</th>
      <th>hours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.996263</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>5.008105</td>
      <td>1</td>
      <td>1.783274</td>
      <td>-0.994983</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.996263</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>0.989913</td>
      <td>1</td>
      <td>-0.269825</td>
      <td>-0.994983</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.996251</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>5.936641</td>
      <td>1</td>
      <td>4.983721</td>
      <td>-0.994972</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.996251</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>4.816249</td>
      <td>1</td>
      <td>1.418291</td>
      <td>-0.994972</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.996239</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>4.248367</td>
      <td>1</td>
      <td>0.670579</td>
      <td>-0.994960</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>277324</th>
      <td>1.034459</td>
      <td>-11.881118</td>
      <td>10.071785</td>
      <td>-9.834783</td>
      <td>-2.066656</td>
      <td>-5.364473</td>
      <td>-2.606837</td>
      <td>-4.918215</td>
      <td>7.305334</td>
      <td>1.914428</td>
      <td>...</td>
      <td>1.436807</td>
      <td>0.250034</td>
      <td>0.943651</td>
      <td>0.823731</td>
      <td>-0.260067</td>
      <td>1</td>
      <td>-0.296653</td>
      <td>1.034951</td>
      <td>59</td>
      <td>23</td>
    </tr>
    <tr>
      <th>277325</th>
      <td>1.034471</td>
      <td>-0.732789</td>
      <td>-0.055080</td>
      <td>2.035030</td>
      <td>-0.738589</td>
      <td>0.868229</td>
      <td>1.058415</td>
      <td>0.024330</td>
      <td>0.294869</td>
      <td>0.584800</td>
      <td>...</td>
      <td>-0.606624</td>
      <td>-0.395255</td>
      <td>0.068472</td>
      <td>-0.053527</td>
      <td>3.210481</td>
      <td>1</td>
      <td>0.038986</td>
      <td>1.034963</td>
      <td>59</td>
      <td>23</td>
    </tr>
    <tr>
      <th>277326</th>
      <td>1.034483</td>
      <td>1.919565</td>
      <td>-0.301254</td>
      <td>-3.249640</td>
      <td>-0.557828</td>
      <td>2.630515</td>
      <td>3.031260</td>
      <td>-0.296827</td>
      <td>0.708417</td>
      <td>0.432454</td>
      <td>...</td>
      <td>0.265745</td>
      <td>-0.087371</td>
      <td>0.004455</td>
      <td>-0.026561</td>
      <td>4.217756</td>
      <td>1</td>
      <td>0.641096</td>
      <td>1.034975</td>
      <td>59</td>
      <td>23</td>
    </tr>
    <tr>
      <th>277327</th>
      <td>1.034483</td>
      <td>-0.240440</td>
      <td>0.530483</td>
      <td>0.702510</td>
      <td>0.689799</td>
      <td>-0.377961</td>
      <td>0.623708</td>
      <td>-0.686180</td>
      <td>0.679145</td>
      <td>0.392087</td>
      <td>...</td>
      <td>-0.569159</td>
      <td>0.546668</td>
      <td>0.108821</td>
      <td>0.104533</td>
      <td>2.302685</td>
      <td>1</td>
      <td>-0.167680</td>
      <td>1.034975</td>
      <td>59</td>
      <td>23</td>
    </tr>
    <tr>
      <th>277328</th>
      <td>1.034530</td>
      <td>-0.533413</td>
      <td>-0.189733</td>
      <td>0.703337</td>
      <td>-0.506271</td>
      <td>-0.012546</td>
      <td>-0.649617</td>
      <td>1.577006</td>
      <td>-0.414650</td>
      <td>0.486180</td>
      <td>...</td>
      <td>-0.473649</td>
      <td>-0.818267</td>
      <td>-0.002415</td>
      <td>0.013649</td>
      <td>5.379902</td>
      <td>1</td>
      <td>2.724796</td>
      <td>1.035022</td>
      <td>59</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
<p>277329 rows × 35 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-27bd2f02-cc2a-41fa-be0d-0c4787c47133')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-27bd2f02-cc2a-41fa-be0d-0c4787c47133 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-27bd2f02-cc2a-41fa-be0d-0c4787c47133');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .markdown id="YMak4PYTYyjF"}
Handling Class Imbalance
:::

::: {.cell .code execution_count="69" id="eHIMlG4oogXy"}
``` {.python}
X = data.drop(['Class','hours','mins'], 1)
Y = data.Class
```
:::

::: {.cell .code execution_count="70" colab="{\"height\":487,\"base_uri\":\"https://localhost:8080/\"}" id="VeOCVVytoxTG" outputId="570c5711-1fa3-4afa-fb0e-7a5282d772ec"}
``` {.python}
X
```

::: {.output .execute_result execution_count="70"}
```{=html}
  <div id="df-0411b51d-315d-40ac-85e1-2420e20cd0de">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.366594</td>
      <td>0.122040</td>
      <td>0.889577</td>
      <td>-0.557672</td>
      <td>-0.739899</td>
      <td>1.076581</td>
      <td>-0.251537</td>
      <td>0.768230</td>
      <td>0.150525</td>
      <td>-0.109897</td>
      <td>...</td>
      <td>-0.041285</td>
      <td>-0.321946</td>
      <td>-0.859182</td>
      <td>0.055476</td>
      <td>0.085934</td>
      <td>-0.457265</td>
      <td>0.126185</td>
      <td>0.214236</td>
      <td>0.067386</td>
      <td>0.582774</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.630280</td>
      <td>1.983664</td>
      <td>-0.144566</td>
      <td>-1.601533</td>
      <td>0.650530</td>
      <td>-0.058817</td>
      <td>-0.920016</td>
      <td>-0.079494</td>
      <td>-0.052650</td>
      <td>0.791831</td>
      <td>...</td>
      <td>-0.315038</td>
      <td>0.101805</td>
      <td>0.502895</td>
      <td>0.007543</td>
      <td>-0.045930</td>
      <td>0.086611</td>
      <td>0.712647</td>
      <td>-0.066077</td>
      <td>-0.058296</td>
      <td>1.747633</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.310533</td>
      <td>-3.095610</td>
      <td>2.341266</td>
      <td>0.978825</td>
      <td>1.302911</td>
      <td>-1.550850</td>
      <td>0.900366</td>
      <td>-1.023949</td>
      <td>1.546270</td>
      <td>1.244595</td>
      <td>...</td>
      <td>0.309719</td>
      <td>-0.354588</td>
      <td>-0.455957</td>
      <td>0.074944</td>
      <td>0.194447</td>
      <td>0.033323</td>
      <td>-0.509168</td>
      <td>-0.694157</td>
      <td>-0.489850</td>
      <td>1.735365</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.078180</td>
      <td>-0.677805</td>
      <td>-0.323111</td>
      <td>1.795775</td>
      <td>-2.280477</td>
      <td>-0.424261</td>
      <td>-1.015679</td>
      <td>0.842032</td>
      <td>-0.386281</td>
      <td>-1.098708</td>
      <td>...</td>
      <td>0.334034</td>
      <td>0.141367</td>
      <td>0.139091</td>
      <td>0.226603</td>
      <td>0.365986</td>
      <td>-0.261502</td>
      <td>-0.817321</td>
      <td>-0.108201</td>
      <td>-0.071991</td>
      <td>4.836290</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.866347</td>
      <td>1.365524</td>
      <td>-0.193916</td>
      <td>0.327665</td>
      <td>-0.513938</td>
      <td>-0.639434</td>
      <td>-0.543843</td>
      <td>-0.733840</td>
      <td>-0.151407</td>
      <td>0.173629</td>
      <td>...</td>
      <td>0.159146</td>
      <td>-0.050360</td>
      <td>0.029099</td>
      <td>-0.056541</td>
      <td>-0.104268</td>
      <td>0.386347</td>
      <td>-0.321959</td>
      <td>0.002022</td>
      <td>0.025594</td>
      <td>2.769522</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>277324</th>
      <td>-0.205604</td>
      <td>1.145484</td>
      <td>-0.781774</td>
      <td>0.488417</td>
      <td>-0.859133</td>
      <td>-0.543703</td>
      <td>1.031799</td>
      <td>-0.869260</td>
      <td>0.403027</td>
      <td>2.153869</td>
      <td>...</td>
      <td>-0.060065</td>
      <td>0.043209</td>
      <td>0.565356</td>
      <td>-0.340819</td>
      <td>-1.081764</td>
      <td>0.825123</td>
      <td>-0.393337</td>
      <td>0.113609</td>
      <td>0.009240</td>
      <td>3.610945</td>
    </tr>
    <tr>
      <th>277325</th>
      <td>-0.826282</td>
      <td>1.205584</td>
      <td>-0.018865</td>
      <td>0.851469</td>
      <td>0.271141</td>
      <td>-0.611746</td>
      <td>-0.355526</td>
      <td>-0.492331</td>
      <td>-0.082980</td>
      <td>1.680204</td>
      <td>...</td>
      <td>-0.092231</td>
      <td>-0.035527</td>
      <td>0.212023</td>
      <td>0.011242</td>
      <td>0.113186</td>
      <td>0.146389</td>
      <td>1.059321</td>
      <td>-0.070533</td>
      <td>0.003090</td>
      <td>2.704778</td>
    </tr>
    <tr>
      <th>277326</th>
      <td>0.635616</td>
      <td>1.823779</td>
      <td>-0.790119</td>
      <td>-0.079887</td>
      <td>0.486926</td>
      <td>-0.992713</td>
      <td>0.013298</td>
      <td>-0.943318</td>
      <td>0.249591</td>
      <td>1.273188</td>
      <td>...</td>
      <td>-0.134108</td>
      <td>0.075700</td>
      <td>0.139171</td>
      <td>0.226039</td>
      <td>-0.361329</td>
      <td>-0.592829</td>
      <td>0.166075</td>
      <td>-0.015923</td>
      <td>-0.039252</td>
      <td>4.246794</td>
    </tr>
    <tr>
      <th>277327</th>
      <td>-0.424570</td>
      <td>-3.729045</td>
      <td>-3.345294</td>
      <td>0.504541</td>
      <td>1.349409</td>
      <td>2.587102</td>
      <td>-2.875825</td>
      <td>-1.848069</td>
      <td>0.775216</td>
      <td>0.299077</td>
      <td>...</td>
      <td>0.889643</td>
      <td>0.444018</td>
      <td>-0.123563</td>
      <td>0.036287</td>
      <td>0.551506</td>
      <td>-0.742764</td>
      <td>0.095890</td>
      <td>0.128720</td>
      <td>-0.504839</td>
      <td>2.510493</td>
    </tr>
    <tr>
      <th>277328</th>
      <td>0.906330</td>
      <td>0.245442</td>
      <td>0.757524</td>
      <td>0.069692</td>
      <td>-0.675437</td>
      <td>0.834176</td>
      <td>-0.494112</td>
      <td>1.009999</td>
      <td>-0.188276</td>
      <td>-0.293707</td>
      <td>...</td>
      <td>-0.070872</td>
      <td>-0.212164</td>
      <td>-0.547161</td>
      <td>0.116291</td>
      <td>0.624563</td>
      <td>-0.456012</td>
      <td>0.097241</td>
      <td>-0.027043</td>
      <td>0.000804</td>
      <td>2.507239</td>
    </tr>
  </tbody>
</table>
<p>277329 rows × 30 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-0411b51d-315d-40ac-85e1-2420e20cd0de')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-0411b51d-315d-40ac-85e1-2420e20cd0de button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-0411b51d-315d-40ac-85e1-2420e20cd0de');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .markdown id="I5nJ-GiNZbkJ"}
Original dataset vs Resampled Dataset
:::

::: {.cell .code execution_count="71" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="HMBA7RADoulG" outputId="f1943bec-bbed-43a9-d545-86245ad2150e"}
``` {.python}
print(f'Original dataset shape : {Counter(Y)}')
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, Y)
print(f'Resampled dataset shape {Counter(y_res)}')
```

::: {.output .stream .stdout}
    Original dataset shape : Counter({1: 277080, -1: 249})
    Resampled dataset shape Counter({1: 277080, -1: 277080})
:::
:::

::: {.cell .code execution_count="72" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="6H852XSAU8q4" outputId="40886845-d1f4-40c1-a4bb-0738c90dd1b3"}
``` {.python}
y_res
```

::: {.output .execute_result execution_count="72"}
    0         1
    1         1
    2         1
    3         1
    4         1
             ..
    554155   -1
    554156   -1
    554157   -1
    554158   -1
    554159   -1
    Name: Class, Length: 554160, dtype: int64
:::
:::

::: {.cell .code execution_count="73" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="SaQqmjPWp13t" outputId="9b31e471-7865-4d6d-9385-aaf8d3db827a"}
``` {.python}
#Dimensionality Reduction and Clustering
# creating a random sample of 5000 points 
X_vis = X_res.sample(1000)
y_vis = y_res.sample(1000) 
print(X_vis.shape)
print(y_vis.shape)
y_vis
```

::: {.output .stream .stdout}
    (1000, 30)
    (1000,)
:::

::: {.output .execute_result execution_count="73"}
    255118    1
    92237     1
    528358   -1
    460015   -1
    211816    1
             ..
    242409    1
    109819    1
    423955   -1
    180758    1
    464455   -1
    Name: Class, Length: 1000, dtype: int64
:::
:::

::: {.cell .code execution_count="74" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="4Y4IN8TwM75Y" outputId="69855a2a-ee0e-40ca-b6da-67d53bc2eda5"}
``` {.python}
y_res
```

::: {.output .execute_result execution_count="74"}
    0         1
    1         1
    2         1
    3         1
    4         1
             ..
    554155   -1
    554156   -1
    554157   -1
    554158   -1
    554159   -1
    Name: Class, Length: 554160, dtype: int64
:::
:::

::: {.cell .code execution_count="75" colab="{\"height\":487,\"base_uri\":\"https://localhost:8080/\"}" id="qumtgYMAEYbH" outputId="c478202e-fb00-4147-b7a2-7432072cd5a7"}
``` {.python}
new_df
```

::: {.output .execute_result execution_count="75"}
```{=html}
  <div id="df-6b7fec24-5886-4e0c-91bb-2a191a15eaf6">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>146594</th>
      <td>87771.0</td>
      <td>-1.350426</td>
      <td>1.390405</td>
      <td>-0.167461</td>
      <td>-2.662750</td>
      <td>0.441095</td>
      <td>-1.109427</td>
      <td>0.836490</td>
      <td>0.343835</td>
      <td>0.413690</td>
      <td>...</td>
      <td>-0.326158</td>
      <td>-0.959353</td>
      <td>-0.035024</td>
      <td>0.553007</td>
      <td>0.052678</td>
      <td>0.361641</td>
      <td>0.302744</td>
      <td>0.225979</td>
      <td>1.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>46909</th>
      <td>42985.0</td>
      <td>-4.075975</td>
      <td>0.963031</td>
      <td>-5.076070</td>
      <td>4.955963</td>
      <td>-0.161437</td>
      <td>-2.832663</td>
      <td>-7.619765</td>
      <td>1.618895</td>
      <td>-2.992092</td>
      <td>...</td>
      <td>1.030738</td>
      <td>0.165328</td>
      <td>-1.017502</td>
      <td>-0.477983</td>
      <td>-0.304987</td>
      <td>-0.106089</td>
      <td>1.899714</td>
      <td>0.511462</td>
      <td>1.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>161968</th>
      <td>114704.0</td>
      <td>-0.940270</td>
      <td>-0.105899</td>
      <td>1.053156</td>
      <td>-0.182287</td>
      <td>0.423150</td>
      <td>-0.790311</td>
      <td>0.380002</td>
      <td>-0.190759</td>
      <td>-1.355259</td>
      <td>...</td>
      <td>-0.090910</td>
      <td>0.336364</td>
      <td>-0.631668</td>
      <td>-0.013696</td>
      <td>1.137620</td>
      <td>-0.173641</td>
      <td>-0.037815</td>
      <td>-0.117793</td>
      <td>62.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>208651</th>
      <td>137211.0</td>
      <td>0.630579</td>
      <td>1.183631</td>
      <td>-5.066283</td>
      <td>2.179903</td>
      <td>-0.703376</td>
      <td>-0.103614</td>
      <td>-3.490350</td>
      <td>1.094734</td>
      <td>-0.717418</td>
      <td>...</td>
      <td>0.621622</td>
      <td>0.043807</td>
      <td>0.102711</td>
      <td>-0.601505</td>
      <td>0.127371</td>
      <td>-0.163009</td>
      <td>0.853792</td>
      <td>0.356503</td>
      <td>39.45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>63421</th>
      <td>50706.0</td>
      <td>-8.461845</td>
      <td>6.866198</td>
      <td>-11.838269</td>
      <td>4.194211</td>
      <td>-6.923097</td>
      <td>-3.221147</td>
      <td>-7.553497</td>
      <td>6.015618</td>
      <td>-2.466143</td>
      <td>...</td>
      <td>0.918244</td>
      <td>-0.715366</td>
      <td>0.210747</td>
      <td>-0.060211</td>
      <td>0.509535</td>
      <td>-0.257284</td>
      <td>1.170027</td>
      <td>0.229301</td>
      <td>99.99</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6774</th>
      <td>8528.0</td>
      <td>0.447396</td>
      <td>2.481954</td>
      <td>-5.660814</td>
      <td>4.455923</td>
      <td>-2.443780</td>
      <td>-2.185040</td>
      <td>-4.716143</td>
      <td>1.249803</td>
      <td>-0.718326</td>
      <td>...</td>
      <td>0.756053</td>
      <td>0.140168</td>
      <td>0.665411</td>
      <td>0.131464</td>
      <td>-1.908217</td>
      <td>0.334808</td>
      <td>0.748534</td>
      <td>0.175414</td>
      <td>1.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>84543</th>
      <td>60353.0</td>
      <td>-3.975216</td>
      <td>0.581573</td>
      <td>-1.880372</td>
      <td>4.319241</td>
      <td>-3.024330</td>
      <td>1.240793</td>
      <td>-1.909559</td>
      <td>0.660718</td>
      <td>-2.752611</td>
      <td>...</td>
      <td>0.578984</td>
      <td>1.397311</td>
      <td>1.045322</td>
      <td>-0.304000</td>
      <td>0.005295</td>
      <td>0.235435</td>
      <td>0.962015</td>
      <td>-0.673557</td>
      <td>454.82</td>
      <td>1</td>
    </tr>
    <tr>
      <th>140152</th>
      <td>83569.0</td>
      <td>0.975148</td>
      <td>-0.435525</td>
      <td>-1.485782</td>
      <td>0.076614</td>
      <td>2.091899</td>
      <td>3.351847</td>
      <td>-0.137473</td>
      <td>0.720743</td>
      <td>-0.173125</td>
      <td>...</td>
      <td>0.076385</td>
      <td>-0.264479</td>
      <td>-0.283245</td>
      <td>1.012633</td>
      <td>0.786117</td>
      <td>-0.330769</td>
      <td>-0.015467</td>
      <td>0.042826</td>
      <td>172.40</td>
      <td>0</td>
    </tr>
    <tr>
      <th>239499</th>
      <td>150138.0</td>
      <td>-2.150855</td>
      <td>2.187917</td>
      <td>-3.430516</td>
      <td>0.119476</td>
      <td>-0.173210</td>
      <td>0.290700</td>
      <td>-2.808988</td>
      <td>-2.679351</td>
      <td>-0.556685</td>
      <td>...</td>
      <td>-0.073205</td>
      <td>0.561496</td>
      <td>-0.075034</td>
      <td>-0.437619</td>
      <td>0.353841</td>
      <td>-0.521339</td>
      <td>0.144465</td>
      <td>0.026588</td>
      <td>50.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9252</th>
      <td>13323.0</td>
      <td>-5.454362</td>
      <td>8.287421</td>
      <td>-12.752811</td>
      <td>8.594342</td>
      <td>-3.106002</td>
      <td>-3.179949</td>
      <td>-9.252794</td>
      <td>4.245062</td>
      <td>-6.329801</td>
      <td>...</td>
      <td>1.846165</td>
      <td>-0.267172</td>
      <td>-0.310804</td>
      <td>-1.201685</td>
      <td>1.352176</td>
      <td>0.608425</td>
      <td>1.574715</td>
      <td>0.808725</td>
      <td>1.00</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>984 rows × 31 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-6b7fec24-5886-4e0c-91bb-2a191a15eaf6')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-6b7fec24-5886-4e0c-91bb-2a191a15eaf6 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-6b7fec24-5886-4e0c-91bb-2a191a15eaf6');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code execution_count="76" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="pJRtqgjZ-3fQ" outputId="3bdaa21e-a257-4055-826b-8ba82bd523a2"}
``` {.python}
X_reduced_tsne = TSNE(n_components=2,
                      random_state=42 , 
                      verbose=2).fit_transform(X_vis.values)
                      
```

::: {.output .stream .stdout}
    [t-SNE] Computing 91 nearest neighbors...
    [t-SNE] Indexed 1000 samples in 0.001s...
    [t-SNE] Computed neighbors for 1000 samples in 0.068s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 1000
    [t-SNE] Mean sigma: 2.708681
    [t-SNE] Computed conditional probabilities in 0.087s
    [t-SNE] Iteration 50: error = 62.3618393, gradient norm = 0.2700375 (50 iterations in 0.493s)
    [t-SNE] Iteration 100: error = 60.1954193, gradient norm = 0.2402890 (50 iterations in 0.396s)
    [t-SNE] Iteration 150: error = 59.2635651, gradient norm = 0.2273580 (50 iterations in 0.402s)
    [t-SNE] Iteration 200: error = 59.4396362, gradient norm = 0.2260049 (50 iterations in 0.405s)
    [t-SNE] Iteration 250: error = 59.0743866, gradient norm = 0.2137973 (50 iterations in 0.416s)
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 59.074387
    [t-SNE] Iteration 300: error = 0.6892751, gradient norm = 0.0007488 (50 iterations in 0.317s)
    [t-SNE] Iteration 350: error = 0.6141629, gradient norm = 0.0003002 (50 iterations in 0.353s)
    [t-SNE] Iteration 400: error = 0.5977510, gradient norm = 0.0002378 (50 iterations in 0.333s)
    [t-SNE] Iteration 450: error = 0.5906724, gradient norm = 0.0001954 (50 iterations in 0.313s)
    [t-SNE] Iteration 500: error = 0.5865698, gradient norm = 0.0001582 (50 iterations in 0.319s)
    [t-SNE] Iteration 550: error = 0.5831249, gradient norm = 0.0001546 (50 iterations in 0.323s)
    [t-SNE] Iteration 600: error = 0.5810931, gradient norm = 0.0001391 (50 iterations in 0.325s)
    [t-SNE] Iteration 650: error = 0.5797158, gradient norm = 0.0001278 (50 iterations in 0.310s)
    [t-SNE] Iteration 700: error = 0.5782769, gradient norm = 0.0001386 (50 iterations in 0.312s)
    [t-SNE] Iteration 750: error = 0.5772893, gradient norm = 0.0001489 (50 iterations in 0.333s)
    [t-SNE] Iteration 800: error = 0.5762801, gradient norm = 0.0001158 (50 iterations in 0.305s)
    [t-SNE] Iteration 850: error = 0.5755168, gradient norm = 0.0001318 (50 iterations in 0.308s)
    [t-SNE] Iteration 900: error = 0.5748398, gradient norm = 0.0001105 (50 iterations in 0.334s)
    [t-SNE] Iteration 950: error = 0.5738946, gradient norm = 0.0001224 (50 iterations in 0.305s)
    [t-SNE] Iteration 1000: error = 0.5731982, gradient norm = 0.0001134 (50 iterations in 0.317s)
    [t-SNE] KL divergence after 1000 iterations: 0.573198
:::
:::

::: {.cell .code execution_count="77" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="6ABBOi0jY4zZ" outputId="c76499ef-930f-4989-f903-db450b78e075"}
``` {.python}
X_reduced_tsne
```

::: {.output .execute_result execution_count="77"}
    array([[ 14.077919 ,   7.5424986],
           [-28.975388 ,  31.206078 ],
           [-34.46293  , -35.13756  ],
           ...,
           [  9.761182 ,  -9.103675 ],
           [ 17.893183 ,   4.865453 ],
           [-25.2198   , -12.656076 ]], dtype=float32)
:::
:::

::: {.cell .code execution_count="78" colab="{\"height\":813,\"base_uri\":\"https://localhost:8080/\"}" id="3tuDuAJVXWfk" outputId="1e8b6286-20c2-4b42-88b5-367db74574b0"}
``` {.python}
# t-SNE scatter plot
f, ax = plt.subplots(figsize=(24,16))

blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')

ax.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y_vis == 0), cmap='coolwarm', label='No Fraud', linewidths=1)
ax.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y_vis == 1), cmap='coolwarm', label='Fraud', linewidths=1)
ax.set_title('t-SNE', fontsize=14)

ax.grid(True)
ax.legend(handles=[blue_patch, red_patch])
```

::: {.output .execute_result execution_count="78"}
    <matplotlib.legend.Legend at 0x7f14404c0cd0>
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/29d44975d3e5d1e3b14dba8a32d452ca6518a5ec.png)
:::
:::

::: {.cell .markdown id="Bzk9Sw06wED6"}
# Training Models
:::

::: {.cell .code execution_count="79" colab="{\"height\":299,\"base_uri\":\"https://localhost:8080/\"}" id="8Z4EXGhJnniP" outputId="ae3e36f6-0af3-4039-a5fc-ee682bebb239"}
``` {.python}
# Scale "Time" and "Amount"
df['scaled_amount'] = RobustScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = RobustScaler().fit_transform(df['Time'].values.reshape(-1,1))

# Make a new dataset named "df_scaled" dropping out original "Time" and "Amount"
df_scaled = df.drop(['Time','Amount'],axis = 1,inplace=False)
df_scaled.head()
```

::: {.output .execute_result execution_count="79"}
```{=html}
  <div id="df-9581316f-f829-4cd3-ae75-24bc5357ea79">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>...</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Class</th>
      <th>mins</th>
      <th>hours</th>
      <th>scaled_amount</th>
      <th>scaled_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>163467</th>
      <td>0.122040</td>
      <td>0.889577</td>
      <td>-0.557672</td>
      <td>-0.739899</td>
      <td>1.076581</td>
      <td>-0.251537</td>
      <td>0.768230</td>
      <td>0.150525</td>
      <td>-0.109897</td>
      <td>-0.638497</td>
      <td>...</td>
      <td>0.085934</td>
      <td>-0.457265</td>
      <td>0.126185</td>
      <td>0.214236</td>
      <td>0.067386</td>
      <td>0</td>
      <td>12</td>
      <td>8</td>
      <td>-0.282401</td>
      <td>0.367345</td>
    </tr>
    <tr>
      <th>211403</th>
      <td>1.983664</td>
      <td>-0.144566</td>
      <td>-1.601533</td>
      <td>0.650530</td>
      <td>-0.058817</td>
      <td>-0.920016</td>
      <td>-0.079494</td>
      <td>-0.052650</td>
      <td>0.791831</td>
      <td>-0.250719</td>
      <td>...</td>
      <td>-0.045930</td>
      <td>0.086611</td>
      <td>0.712647</td>
      <td>-0.066077</td>
      <td>-0.058296</td>
      <td>0</td>
      <td>26</td>
      <td>14</td>
      <td>-0.227206</td>
      <td>0.630928</td>
    </tr>
    <tr>
      <th>80125</th>
      <td>-3.095610</td>
      <td>2.341266</td>
      <td>0.978825</td>
      <td>1.302911</td>
      <td>-1.550850</td>
      <td>0.900366</td>
      <td>-1.023949</td>
      <td>1.546270</td>
      <td>1.244595</td>
      <td>1.157214</td>
      <td>...</td>
      <td>0.194447</td>
      <td>0.033323</td>
      <td>-0.509168</td>
      <td>-0.694157</td>
      <td>-0.489850</td>
      <td>0</td>
      <td>12</td>
      <td>16</td>
      <td>-0.228184</td>
      <td>-0.309520</td>
    </tr>
    <tr>
      <th>126826</th>
      <td>-0.677805</td>
      <td>-0.323111</td>
      <td>1.795775</td>
      <td>-2.280477</td>
      <td>-0.424261</td>
      <td>-1.015679</td>
      <td>0.842032</td>
      <td>-0.386281</td>
      <td>-1.098708</td>
      <td>-0.234947</td>
      <td>...</td>
      <td>0.365986</td>
      <td>-0.261502</td>
      <td>-0.817321</td>
      <td>-0.108201</td>
      <td>-0.071991</td>
      <td>0</td>
      <td>41</td>
      <td>21</td>
      <td>1.453224</td>
      <td>-0.077257</td>
    </tr>
    <tr>
      <th>8221</th>
      <td>1.365524</td>
      <td>-0.193916</td>
      <td>0.327665</td>
      <td>-0.513938</td>
      <td>-0.639434</td>
      <td>-0.543843</td>
      <td>-0.733840</td>
      <td>-0.151407</td>
      <td>0.173629</td>
      <td>-0.002654</td>
      <td>...</td>
      <td>-0.104268</td>
      <td>0.386347</td>
      <td>-0.321959</td>
      <td>0.002022</td>
      <td>0.025594</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>-0.084539</td>
      <td>-0.865118</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-9581316f-f829-4cd3-ae75-24bc5357ea79')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-9581316f-f829-4cd3-ae75-24bc5357ea79 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9581316f-f829-4cd3-ae75-24bc5357ea79');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code execution_count="82" id="lbavBVEEZ1WD"}
``` {.python}
#Separating X and Y
X = df_scaled.drop(['Class', 'mins', 'hours'], axis=1)
y = df_scaled['Class']
```
:::

::: {.cell .code execution_count="83" colab="{\"height\":487,\"base_uri\":\"https://localhost:8080/\"}" id="l7TOqaEnZ50g" outputId="4a316b40-bc3e-4a88-fac1-365a7519ee65"}
``` {.python}
X
```

::: {.output .execute_result execution_count="83"}
```{=html}
  <div id="df-1984218d-afa2-4a60-8893-1e5834b859b0">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>scaled_amount</th>
      <th>scaled_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>163467</th>
      <td>0.122040</td>
      <td>0.889577</td>
      <td>-0.557672</td>
      <td>-0.739899</td>
      <td>1.076581</td>
      <td>-0.251537</td>
      <td>0.768230</td>
      <td>0.150525</td>
      <td>-0.109897</td>
      <td>-0.638497</td>
      <td>...</td>
      <td>-0.321946</td>
      <td>-0.859182</td>
      <td>0.055476</td>
      <td>0.085934</td>
      <td>-0.457265</td>
      <td>0.126185</td>
      <td>0.214236</td>
      <td>0.067386</td>
      <td>-0.282401</td>
      <td>0.367345</td>
    </tr>
    <tr>
      <th>211403</th>
      <td>1.983664</td>
      <td>-0.144566</td>
      <td>-1.601533</td>
      <td>0.650530</td>
      <td>-0.058817</td>
      <td>-0.920016</td>
      <td>-0.079494</td>
      <td>-0.052650</td>
      <td>0.791831</td>
      <td>-0.250719</td>
      <td>...</td>
      <td>0.101805</td>
      <td>0.502895</td>
      <td>0.007543</td>
      <td>-0.045930</td>
      <td>0.086611</td>
      <td>0.712647</td>
      <td>-0.066077</td>
      <td>-0.058296</td>
      <td>-0.227206</td>
      <td>0.630928</td>
    </tr>
    <tr>
      <th>80125</th>
      <td>-3.095610</td>
      <td>2.341266</td>
      <td>0.978825</td>
      <td>1.302911</td>
      <td>-1.550850</td>
      <td>0.900366</td>
      <td>-1.023949</td>
      <td>1.546270</td>
      <td>1.244595</td>
      <td>1.157214</td>
      <td>...</td>
      <td>-0.354588</td>
      <td>-0.455957</td>
      <td>0.074944</td>
      <td>0.194447</td>
      <td>0.033323</td>
      <td>-0.509168</td>
      <td>-0.694157</td>
      <td>-0.489850</td>
      <td>-0.228184</td>
      <td>-0.309520</td>
    </tr>
    <tr>
      <th>126826</th>
      <td>-0.677805</td>
      <td>-0.323111</td>
      <td>1.795775</td>
      <td>-2.280477</td>
      <td>-0.424261</td>
      <td>-1.015679</td>
      <td>0.842032</td>
      <td>-0.386281</td>
      <td>-1.098708</td>
      <td>-0.234947</td>
      <td>...</td>
      <td>0.141367</td>
      <td>0.139091</td>
      <td>0.226603</td>
      <td>0.365986</td>
      <td>-0.261502</td>
      <td>-0.817321</td>
      <td>-0.108201</td>
      <td>-0.071991</td>
      <td>1.453224</td>
      <td>-0.077257</td>
    </tr>
    <tr>
      <th>8221</th>
      <td>1.365524</td>
      <td>-0.193916</td>
      <td>0.327665</td>
      <td>-0.513938</td>
      <td>-0.639434</td>
      <td>-0.543843</td>
      <td>-0.733840</td>
      <td>-0.151407</td>
      <td>0.173629</td>
      <td>-0.002654</td>
      <td>...</td>
      <td>-0.050360</td>
      <td>0.029099</td>
      <td>-0.056541</td>
      <td>-0.104268</td>
      <td>0.386347</td>
      <td>-0.321959</td>
      <td>0.002022</td>
      <td>0.025594</td>
      <td>-0.084539</td>
      <td>-0.865118</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9693</th>
      <td>1.205584</td>
      <td>-0.018865</td>
      <td>0.851469</td>
      <td>0.271141</td>
      <td>-0.611746</td>
      <td>-0.355526</td>
      <td>-0.492331</td>
      <td>-0.082980</td>
      <td>1.680204</td>
      <td>-0.574047</td>
      <td>...</td>
      <td>-0.035527</td>
      <td>0.212023</td>
      <td>0.011242</td>
      <td>0.113186</td>
      <td>0.146389</td>
      <td>1.059321</td>
      <td>-0.070533</td>
      <td>0.003090</td>
      <td>-0.098512</td>
      <td>-0.825068</td>
    </tr>
    <tr>
      <th>212424</th>
      <td>1.823779</td>
      <td>-0.790119</td>
      <td>-0.079887</td>
      <td>0.486926</td>
      <td>-0.992713</td>
      <td>0.013298</td>
      <td>-0.943318</td>
      <td>0.249591</td>
      <td>1.273188</td>
      <td>0.203302</td>
      <td>...</td>
      <td>0.075700</td>
      <td>0.139171</td>
      <td>0.226039</td>
      <td>-0.361329</td>
      <td>-0.592829</td>
      <td>0.166075</td>
      <td>-0.015923</td>
      <td>-0.039252</td>
      <td>0.669042</td>
      <td>0.636262</td>
    </tr>
    <tr>
      <th>59004</th>
      <td>-3.729045</td>
      <td>-3.345294</td>
      <td>0.504541</td>
      <td>1.349409</td>
      <td>2.587102</td>
      <td>-2.875825</td>
      <td>-1.848069</td>
      <td>0.775216</td>
      <td>0.299077</td>
      <td>-1.420333</td>
      <td>...</td>
      <td>0.444018</td>
      <td>-0.123563</td>
      <td>0.036287</td>
      <td>0.551506</td>
      <td>-0.742764</td>
      <td>0.095890</td>
      <td>0.128720</td>
      <td>-0.504839</td>
      <td>-0.135401</td>
      <td>-0.423513</td>
    </tr>
    <tr>
      <th>265434</th>
      <td>0.245442</td>
      <td>0.757524</td>
      <td>0.069692</td>
      <td>-0.675437</td>
      <td>0.834176</td>
      <td>-0.494112</td>
      <td>1.009999</td>
      <td>-0.188276</td>
      <td>-0.293707</td>
      <td>-0.637338</td>
      <td>...</td>
      <td>-0.212164</td>
      <td>-0.547161</td>
      <td>0.116291</td>
      <td>0.624563</td>
      <td>-0.456012</td>
      <td>0.097241</td>
      <td>-0.027043</td>
      <td>0.000804</td>
      <td>-0.135960</td>
      <td>0.906872</td>
    </tr>
    <tr>
      <th>92081</th>
      <td>-1.810611</td>
      <td>-6.514130</td>
      <td>-0.454912</td>
      <td>0.306467</td>
      <td>-3.931548</td>
      <td>-0.132263</td>
      <td>0.750604</td>
      <td>-0.541343</td>
      <td>-1.549850</td>
      <td>0.502068</td>
      <td>...</td>
      <td>1.010159</td>
      <td>-0.306694</td>
      <td>-1.448084</td>
      <td>1.042912</td>
      <td>-0.303816</td>
      <td>-0.260250</td>
      <td>-0.233200</td>
      <td>0.330292</td>
      <td>23.287221</td>
      <td>-0.245832</td>
    </tr>
  </tbody>
</table>
<p>284807 rows × 30 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1984218d-afa2-4a60-8893-1e5834b859b0')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-1984218d-afa2-4a60-8893-1e5834b859b0 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1984218d-afa2-4a60-8893-1e5834b859b0');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code execution_count="84" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="QJ27UIiSZ_Pr" outputId="89f4afdf-1591-4cf7-f0c3-d6ee0abb9fe3"}
``` {.python}
y
```

::: {.output .execute_result execution_count="84"}
    163467    0
    211403    0
    80125     0
    126826    0
    8221      0
             ..
    9693      0
    212424    0
    59004     0
    265434    0
    92081     0
    Name: Class, Length: 284807, dtype: int64
:::
:::

::: {.cell .markdown id="l0uFvyZ0Jo8T"}
# **Logistic Regression**
:::

::: {.cell .code execution_count="85" id="iCW9DGiEPqlz"}
``` {.python}
#Train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.20, random_state = 0)
```
:::

::: {.cell .code execution_count="86" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="B4WoMZ-cP-r7" outputId="844d7e13-ea2c-429f-ea99-68cceb730f9a"}
``` {.python}
model = LogisticRegression(random_state=1)
model.fit(X_train, y_train)
model.predict_proba(X_test)
```

::: {.output .execute_result execution_count="86"}
    array([[9.99267697e-01, 7.32302773e-04],
           [9.99988889e-01, 1.11112760e-05],
           [9.99756236e-01, 2.43764211e-04],
           ...,
           [9.99891195e-01, 1.08804720e-04],
           [9.99942084e-01, 5.79156451e-05],
           [9.99615470e-01, 3.84530456e-04]])
:::
:::

::: {.cell .code execution_count="100" id="d8WXsmUCQQHC"}
``` {.python}
y_predicted = model.predict(X_test)
```
:::

::: {.cell .code execution_count="101" colab="{\"height\":585,\"base_uri\":\"https://localhost:8080/\"}" id="YVJJ0b-8QcDZ" outputId="43da0360-9ff4-44a0-f2c2-639e599681f6"}
``` {.python}
def plot_confusion_matrix(cm, classes, title, normalize = False, cmap = plt.cm.Blues):
    title = 'Confusion Matrix of {}'.format(title)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm = confusion_matrix(y_test, y_predicted) # Support Vector Machine
plt.rcParams['figure.figsize'] = (8, 8)
svm_cm_plot = plot_confusion_matrix(cm, 
                                classes = ['0','1'], 
                                normalize = False, title = 'Confusion matrix of Logistic Regression')
plt.savefig('svm_cm_plot.png')
plt.show()
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/14c491fcd168b366e1c689bac6b4cfd48cb15ea5.png)
:::
:::

::: {.cell .code execution_count="102" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="z1H6GN-aQmpa" outputId="49071948-72df-4a15-e747-100bd2d1ed38"}
``` {.python}
print(classification_report(y_test, y_predicted))
```

::: {.output .stream .stdout}
                  precision    recall  f1-score   support

               0       1.00      0.98      0.99     56854
               1       0.09      0.84      0.17       108

        accuracy                           0.98     56962
       macro avg       0.55      0.91      0.58     56962
    weighted avg       1.00      0.98      0.99     56962
:::
:::

::: {.cell .code execution_count="103" colab="{\"height\":780,\"base_uri\":\"https://localhost:8080/\"}" id="26xxh8kWsFUp" outputId="04b7d7c6-63ad-4d23-f8af-8040671908a3"}
``` {.python}
# Create true and false positive rates
false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_predicted)

# Calculate Area Under the Receiver Operating Characteristic Curve 
probs = model.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, probs[:, 1])
print('ROC AUC Score:',roc_auc)

# Obtain precision and recall 
precision, recall, thresholds = precision_recall_curve(y_test, y_predicted)

# Calculate average precision 
average_precision = average_precision_score(y_test, y_predicted)

# Define a roc_curve function
def plot_roc_curve(false_positive_rate,true_positive_rate,roc_auc):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)
    plt.plot([0,1],[0,1], linewidth=5)
    plt.xlim([-0.01, 1])
    plt.ylim([0, 1.01])
    plt.legend(loc='upper right')
    plt.title('Receiver operating characteristic curve (ROC)')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# Define a precision_recall_curve function
def plot_pr_curve(recall, precision, average_precision):
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()

# Print the classifcation report and confusion matrix
print('Classification report:\n', classification_report(y_test, y_predicted))
cm = confusion_matrix(y_test, y_predicted) # Support Vector Machine
plt.rcParams['figure.figsize'] = (8, 8)
svm_cm_plot = plot_confusion_matrix(cm, 
                                classes = ['0','1'], 
                                normalize = False, title = 'Confusion matrix of Logistic Regression')
plt.savefig('svm_cm_plot.png')
plt.show()
```

::: {.output .stream .stdout}
    ROC AUC Score: 0.9529494976737034
    Classification report:
                   precision    recall  f1-score   support

               0       1.00      0.98      0.99     56854
               1       0.09      0.84      0.17       108

        accuracy                           0.98     56962
       macro avg       0.55      0.91      0.58     56962
    weighted avg       1.00      0.98      0.99     56962
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/14c491fcd168b366e1c689bac6b4cfd48cb15ea5.png)
:::
:::

::: {.cell .code execution_count="104" colab="{\"height\":1000,\"base_uri\":\"https://localhost:8080/\"}" id="9kNynAdzr70o" outputId="c1e69745-cc4b-4c5b-eb74-8494d2145f9d"}
``` {.python}
# Calculate average precision 
average_precision = average_precision_score(y_test, y_predicted)

# Obtain precision and recall 
precision, recall, _ = precision_recall_curve(y_test, y_predicted)

# Plot the roc curve 
plot_roc_curve(false_positive_rate,true_positive_rate,roc_auc)

# Plot the recall precision tradeoff
plot_pr_curve(recall, precision, average_precision)
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/ac61c11d31a5b9a4b6b56c8417102d3c1f9aa1f0.png)
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/0e66204dd5f64151d89f4603c75713b1aef7f05b.png)
:::
:::

::: {.cell .code execution_count="106" colab="{\"height\":762,\"base_uri\":\"https://localhost:8080/\"}" id="9j6OPbF4nOPZ" outputId="3967b930-97bb-4478-c338-57f7d017b595"}
``` {.python}
# Create the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

# Define which resampling method and which ML model to use in the pipeline
resampling = BorderlineSMOTE(kind='borderline-2',random_state=0) # instead SMOTE(kind='borderline2') 
model = LogisticRegression() 

# Define the pipeline, tell it to combine SMOTE with the Logistic Regression model
pipeline = Pipeline([('SMOTE', resampling), ('Logistic Regression', model)])

# Fit your pipeline onto your training set and obtain predictions by fitting the model onto the test data 
pipeline.fit(X_train, y_train) 
y_predicted = pipeline.predict(X_test)

# Obtain the results from the classification report and confusion matrix 
print('Classifcation report:\n', classification_report(y_test, y_predicted))

cm = confusion_matrix(y_test, y_predicted) # Support Vector Machine
plt.rcParams['figure.figsize'] = (8, 8)
svm_cm_plot = plot_confusion_matrix(cm, 
                                classes = ['0','1'], 
                                normalize = False, title = 'Confusion matrix of Logistic Regression')
plt.savefig('svm_cm_plot.png')
plt.show()
```

::: {.output .stream .stdout}
    Classifcation report:
                   precision    recall  f1-score   support

               0       1.00      0.98      0.99     56854
               1       0.09      0.84      0.17       108

        accuracy                           0.98     56962
       macro avg       0.55      0.91      0.58     56962
    weighted avg       1.00      0.98      0.99     56962
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/14c491fcd168b366e1c689bac6b4cfd48cb15ea5.png)
:::
:::

::: {.cell .code execution_count="107" colab="{\"height\":1000,\"base_uri\":\"https://localhost:8080/\"}" id="UgAZj5zCsvE7" outputId="01b0e8bb-c486-4864-88cf-3fcc9cb48f59"}
``` {.python}
# Calculate average precision 
average_precision = average_precision_score(y_test, y_predicted)

# Obtain precision and recall 
precision, recall, _ = precision_recall_curve(y_test, y_predicted)

# Plot the roc curve 
plot_roc_curve(false_positive_rate,true_positive_rate,roc_auc)

# Plot the recall precision tradeoff
plot_pr_curve(recall, precision, average_precision)
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/ac61c11d31a5b9a4b6b56c8417102d3c1f9aa1f0.png)
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/0e66204dd5f64151d89f4603c75713b1aef7f05b.png)
:::
:::

::: {.cell .markdown id="hEXdXG1idE4O"}
Training Part of Logistic Regression after removing
\'V1\',\'V2\',\'V3\',\'V7\',\'V8\',\'V9\',\'V10\',\'V15\',\'V17\',\'V21\',\'V22\',\'V23\',\'V24\',\'V25\',\'V26\',\'V27\',\'V28\'
:::

::: {.cell .code execution_count="111" colab="{\"height\":270,\"base_uri\":\"https://localhost:8080/\"}" id="1U4d2O9BZJag" outputId="2e89b1b4-d5b0-4f3f-a707-d0da92835ac6"}
``` {.python}
# Scale "Time" and "Amount"
df['scaled_amount'] = RobustScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = RobustScaler().fit_transform(df['Time'].values.reshape(-1,1))

# Make a new dataset named "df_scaled" dropping out original "Time" and "Amount"
df_scaled = df.drop(['Time','Amount','V1','V2','V3','V7','V8','V9','V10','V15','V17','V21','V22','V23','V24','V25','V26','V27','V28'],axis = 1,inplace=False)
df_scaled.head()
```

::: {.output .execute_result execution_count="111"}
```{=html}
  <div id="df-0521f462-6cba-4167-956f-b0df4925b62d">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V11</th>
      <th>V12</th>
      <th>V13</th>
      <th>V14</th>
      <th>V16</th>
      <th>V18</th>
      <th>V19</th>
      <th>V20</th>
      <th>Class</th>
      <th>mins</th>
      <th>hours</th>
      <th>scaled_amount</th>
      <th>scaled_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>163467</th>
      <td>-0.739899</td>
      <td>1.076581</td>
      <td>-0.251537</td>
      <td>0.539103</td>
      <td>-0.138719</td>
      <td>-1.166615</td>
      <td>-0.608435</td>
      <td>0.628549</td>
      <td>0.448800</td>
      <td>0.133305</td>
      <td>-0.041285</td>
      <td>0</td>
      <td>12</td>
      <td>8</td>
      <td>-0.282401</td>
      <td>0.367345</td>
    </tr>
    <tr>
      <th>211403</th>
      <td>0.650530</td>
      <td>-0.058817</td>
      <td>-0.920016</td>
      <td>0.778779</td>
      <td>-0.075154</td>
      <td>-2.119390</td>
      <td>-0.606401</td>
      <td>0.067119</td>
      <td>0.587165</td>
      <td>0.142988</td>
      <td>-0.315038</td>
      <td>0</td>
      <td>26</td>
      <td>14</td>
      <td>-0.227206</td>
      <td>0.630928</td>
    </tr>
    <tr>
      <th>80125</th>
      <td>1.302911</td>
      <td>-1.550850</td>
      <td>0.900366</td>
      <td>0.377390</td>
      <td>1.326861</td>
      <td>-1.034335</td>
      <td>-0.217089</td>
      <td>-1.368498</td>
      <td>-0.560265</td>
      <td>1.398303</td>
      <td>0.309719</td>
      <td>0</td>
      <td>12</td>
      <td>16</td>
      <td>-0.228184</td>
      <td>-0.309520</td>
    </tr>
    <tr>
      <th>126826</th>
      <td>-2.280477</td>
      <td>-0.424261</td>
      <td>-1.015679</td>
      <td>-0.346110</td>
      <td>-0.435490</td>
      <td>0.866670</td>
      <td>-0.785313</td>
      <td>1.756455</td>
      <td>-1.530292</td>
      <td>-1.088577</td>
      <td>0.334034</td>
      <td>0</td>
      <td>41</td>
      <td>21</td>
      <td>1.453224</td>
      <td>-0.077257</td>
    </tr>
    <tr>
      <th>8221</th>
      <td>-0.513938</td>
      <td>-0.639434</td>
      <td>-0.543843</td>
      <td>2.576528</td>
      <td>-2.135597</td>
      <td>2.817757</td>
      <td>0.225021</td>
      <td>1.783841</td>
      <td>-0.159246</td>
      <td>0.534097</td>
      <td>0.159146</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>-0.084539</td>
      <td>-0.865118</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-0521f462-6cba-4167-956f-b0df4925b62d')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-0521f462-6cba-4167-956f-b0df4925b62d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-0521f462-6cba-4167-956f-b0df4925b62d');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code execution_count="112" id="8wy0QHsJZPPF"}
``` {.python}
#Separating X and Y
X = df_scaled.drop(['Class', 'mins', 'hours'], axis=1)
y = df_scaled['Class']
```
:::

::: {.cell .code execution_count="114" colab="{\"height\":762,\"base_uri\":\"https://localhost:8080/\"}" id="9wzgDR6KZUpS" outputId="9a85b000-972b-49f4-a480-c08c84e6ac67"}
``` {.python}
# Create the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=0)
# Define which resampling method and which ML model to use in the pipeline
resampling = BorderlineSMOTE()
model = LogisticRegression()

# Define the pipeline, tell it to combine SMOTE with the Decision tree model
pipeline = Pipeline([('SMOTE', resampling), ('Decision Tree Classifier', model)])

# Fit your pipeline onto your training set and obtain predictions by fitting the model onto the test data 
pipeline.fit(X_train, y_train) 
y_predicted = pipeline.predict(X_test)

# Obtain the results from the classification report and confusion matrix 
print('Classifcation report:\n', classification_report(y_test, y_predicted))

cm = confusion_matrix(y_test, y_predicted) # Support Vector Machine
plt.rcParams['figure.figsize'] = (8, 8)
svm_cm_plot = plot_confusion_matrix(cm, 
                                classes = ['0','1'], 
                                normalize = False, title = 'Confusion matrix of Logistic Regression')
plt.savefig('svm_cm_plot.png')
plt.show()
```

::: {.output .stream .stdout}
    Classifcation report:
                   precision    recall  f1-score   support

               0       1.00      0.99      1.00     56854
               1       0.15      0.82      0.26       108

        accuracy                           0.99     56962
       macro avg       0.58      0.91      0.63     56962
    weighted avg       1.00      0.99      0.99     56962
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/d2caa1a5364cfc13272ad2fc65941ad38c04c28a.png)
:::
:::

::: {.cell .code execution_count="115" colab="{\"height\":1000,\"base_uri\":\"https://localhost:8080/\"}" id="jprdyCSzZdJ6" outputId="50990e54-f588-44e3-f764-dab2b8d1c92a"}
``` {.python}
# Calculate average precision 
average_precision = average_precision_score(y_test, y_predicted)

# Obtain precision and recall 
precision, recall, _ = precision_recall_curve(y_test, y_predicted)

# Plot the roc curve 
plot_roc_curve(false_positive_rate,true_positive_rate,roc_auc)

# Plot recall precision curve
plot_pr_curve(recall, precision, average_precision)
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/ac61c11d31a5b9a4b6b56c8417102d3c1f9aa1f0.png)
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/959f69ad9982a64fceebb8bd184078e3f660fd7d.png)
:::
:::

::: {.cell .markdown id="BzsrhmE_penQ"}
# Decision Tree
:::

::: {.cell .code execution_count="116" colab="{\"height\":299,\"base_uri\":\"https://localhost:8080/\"}" id="DtY-eTIhsVhi" outputId="a0f1c036-8f65-4d26-b4bf-185eebc5de81"}
``` {.python}
# Scale "Time" and "Amount"
df['scaled_amount'] = RobustScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = RobustScaler().fit_transform(df['Time'].values.reshape(-1,1))

# Make a new dataset named "df_scaled" dropping out original "Time" and "Amount"
df_scaled = df.drop(['Time','Amount'],axis = 1,inplace=False)
df_scaled.head()
```

::: {.output .execute_result execution_count="116"}
```{=html}
  <div id="df-ee4cc56c-a09d-45cd-b70d-a07823141d28">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>...</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Class</th>
      <th>mins</th>
      <th>hours</th>
      <th>scaled_amount</th>
      <th>scaled_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>163467</th>
      <td>0.122040</td>
      <td>0.889577</td>
      <td>-0.557672</td>
      <td>-0.739899</td>
      <td>1.076581</td>
      <td>-0.251537</td>
      <td>0.768230</td>
      <td>0.150525</td>
      <td>-0.109897</td>
      <td>-0.638497</td>
      <td>...</td>
      <td>0.085934</td>
      <td>-0.457265</td>
      <td>0.126185</td>
      <td>0.214236</td>
      <td>0.067386</td>
      <td>0</td>
      <td>12</td>
      <td>8</td>
      <td>-0.282401</td>
      <td>0.367345</td>
    </tr>
    <tr>
      <th>211403</th>
      <td>1.983664</td>
      <td>-0.144566</td>
      <td>-1.601533</td>
      <td>0.650530</td>
      <td>-0.058817</td>
      <td>-0.920016</td>
      <td>-0.079494</td>
      <td>-0.052650</td>
      <td>0.791831</td>
      <td>-0.250719</td>
      <td>...</td>
      <td>-0.045930</td>
      <td>0.086611</td>
      <td>0.712647</td>
      <td>-0.066077</td>
      <td>-0.058296</td>
      <td>0</td>
      <td>26</td>
      <td>14</td>
      <td>-0.227206</td>
      <td>0.630928</td>
    </tr>
    <tr>
      <th>80125</th>
      <td>-3.095610</td>
      <td>2.341266</td>
      <td>0.978825</td>
      <td>1.302911</td>
      <td>-1.550850</td>
      <td>0.900366</td>
      <td>-1.023949</td>
      <td>1.546270</td>
      <td>1.244595</td>
      <td>1.157214</td>
      <td>...</td>
      <td>0.194447</td>
      <td>0.033323</td>
      <td>-0.509168</td>
      <td>-0.694157</td>
      <td>-0.489850</td>
      <td>0</td>
      <td>12</td>
      <td>16</td>
      <td>-0.228184</td>
      <td>-0.309520</td>
    </tr>
    <tr>
      <th>126826</th>
      <td>-0.677805</td>
      <td>-0.323111</td>
      <td>1.795775</td>
      <td>-2.280477</td>
      <td>-0.424261</td>
      <td>-1.015679</td>
      <td>0.842032</td>
      <td>-0.386281</td>
      <td>-1.098708</td>
      <td>-0.234947</td>
      <td>...</td>
      <td>0.365986</td>
      <td>-0.261502</td>
      <td>-0.817321</td>
      <td>-0.108201</td>
      <td>-0.071991</td>
      <td>0</td>
      <td>41</td>
      <td>21</td>
      <td>1.453224</td>
      <td>-0.077257</td>
    </tr>
    <tr>
      <th>8221</th>
      <td>1.365524</td>
      <td>-0.193916</td>
      <td>0.327665</td>
      <td>-0.513938</td>
      <td>-0.639434</td>
      <td>-0.543843</td>
      <td>-0.733840</td>
      <td>-0.151407</td>
      <td>0.173629</td>
      <td>-0.002654</td>
      <td>...</td>
      <td>-0.104268</td>
      <td>0.386347</td>
      <td>-0.321959</td>
      <td>0.002022</td>
      <td>0.025594</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>-0.084539</td>
      <td>-0.865118</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ee4cc56c-a09d-45cd-b70d-a07823141d28')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-ee4cc56c-a09d-45cd-b70d-a07823141d28 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ee4cc56c-a09d-45cd-b70d-a07823141d28');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code execution_count="117" colab="{\"height\":487,\"base_uri\":\"https://localhost:8080/\"}" id="4OsfxoOJtukk" outputId="71bc22ff-3685-4dfa-c29c-01751441ea59"}
``` {.python}
#Separating X and Y
X = df_scaled.drop(['Class', 'mins', 'hours'], axis=1)
y = df_scaled['Class']
X
```

::: {.output .execute_result execution_count="117"}
```{=html}
  <div id="df-f926d18e-cce7-434a-8232-f81059bf2aad">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>scaled_amount</th>
      <th>scaled_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>163467</th>
      <td>0.122040</td>
      <td>0.889577</td>
      <td>-0.557672</td>
      <td>-0.739899</td>
      <td>1.076581</td>
      <td>-0.251537</td>
      <td>0.768230</td>
      <td>0.150525</td>
      <td>-0.109897</td>
      <td>-0.638497</td>
      <td>...</td>
      <td>-0.321946</td>
      <td>-0.859182</td>
      <td>0.055476</td>
      <td>0.085934</td>
      <td>-0.457265</td>
      <td>0.126185</td>
      <td>0.214236</td>
      <td>0.067386</td>
      <td>-0.282401</td>
      <td>0.367345</td>
    </tr>
    <tr>
      <th>211403</th>
      <td>1.983664</td>
      <td>-0.144566</td>
      <td>-1.601533</td>
      <td>0.650530</td>
      <td>-0.058817</td>
      <td>-0.920016</td>
      <td>-0.079494</td>
      <td>-0.052650</td>
      <td>0.791831</td>
      <td>-0.250719</td>
      <td>...</td>
      <td>0.101805</td>
      <td>0.502895</td>
      <td>0.007543</td>
      <td>-0.045930</td>
      <td>0.086611</td>
      <td>0.712647</td>
      <td>-0.066077</td>
      <td>-0.058296</td>
      <td>-0.227206</td>
      <td>0.630928</td>
    </tr>
    <tr>
      <th>80125</th>
      <td>-3.095610</td>
      <td>2.341266</td>
      <td>0.978825</td>
      <td>1.302911</td>
      <td>-1.550850</td>
      <td>0.900366</td>
      <td>-1.023949</td>
      <td>1.546270</td>
      <td>1.244595</td>
      <td>1.157214</td>
      <td>...</td>
      <td>-0.354588</td>
      <td>-0.455957</td>
      <td>0.074944</td>
      <td>0.194447</td>
      <td>0.033323</td>
      <td>-0.509168</td>
      <td>-0.694157</td>
      <td>-0.489850</td>
      <td>-0.228184</td>
      <td>-0.309520</td>
    </tr>
    <tr>
      <th>126826</th>
      <td>-0.677805</td>
      <td>-0.323111</td>
      <td>1.795775</td>
      <td>-2.280477</td>
      <td>-0.424261</td>
      <td>-1.015679</td>
      <td>0.842032</td>
      <td>-0.386281</td>
      <td>-1.098708</td>
      <td>-0.234947</td>
      <td>...</td>
      <td>0.141367</td>
      <td>0.139091</td>
      <td>0.226603</td>
      <td>0.365986</td>
      <td>-0.261502</td>
      <td>-0.817321</td>
      <td>-0.108201</td>
      <td>-0.071991</td>
      <td>1.453224</td>
      <td>-0.077257</td>
    </tr>
    <tr>
      <th>8221</th>
      <td>1.365524</td>
      <td>-0.193916</td>
      <td>0.327665</td>
      <td>-0.513938</td>
      <td>-0.639434</td>
      <td>-0.543843</td>
      <td>-0.733840</td>
      <td>-0.151407</td>
      <td>0.173629</td>
      <td>-0.002654</td>
      <td>...</td>
      <td>-0.050360</td>
      <td>0.029099</td>
      <td>-0.056541</td>
      <td>-0.104268</td>
      <td>0.386347</td>
      <td>-0.321959</td>
      <td>0.002022</td>
      <td>0.025594</td>
      <td>-0.084539</td>
      <td>-0.865118</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9693</th>
      <td>1.205584</td>
      <td>-0.018865</td>
      <td>0.851469</td>
      <td>0.271141</td>
      <td>-0.611746</td>
      <td>-0.355526</td>
      <td>-0.492331</td>
      <td>-0.082980</td>
      <td>1.680204</td>
      <td>-0.574047</td>
      <td>...</td>
      <td>-0.035527</td>
      <td>0.212023</td>
      <td>0.011242</td>
      <td>0.113186</td>
      <td>0.146389</td>
      <td>1.059321</td>
      <td>-0.070533</td>
      <td>0.003090</td>
      <td>-0.098512</td>
      <td>-0.825068</td>
    </tr>
    <tr>
      <th>212424</th>
      <td>1.823779</td>
      <td>-0.790119</td>
      <td>-0.079887</td>
      <td>0.486926</td>
      <td>-0.992713</td>
      <td>0.013298</td>
      <td>-0.943318</td>
      <td>0.249591</td>
      <td>1.273188</td>
      <td>0.203302</td>
      <td>...</td>
      <td>0.075700</td>
      <td>0.139171</td>
      <td>0.226039</td>
      <td>-0.361329</td>
      <td>-0.592829</td>
      <td>0.166075</td>
      <td>-0.015923</td>
      <td>-0.039252</td>
      <td>0.669042</td>
      <td>0.636262</td>
    </tr>
    <tr>
      <th>59004</th>
      <td>-3.729045</td>
      <td>-3.345294</td>
      <td>0.504541</td>
      <td>1.349409</td>
      <td>2.587102</td>
      <td>-2.875825</td>
      <td>-1.848069</td>
      <td>0.775216</td>
      <td>0.299077</td>
      <td>-1.420333</td>
      <td>...</td>
      <td>0.444018</td>
      <td>-0.123563</td>
      <td>0.036287</td>
      <td>0.551506</td>
      <td>-0.742764</td>
      <td>0.095890</td>
      <td>0.128720</td>
      <td>-0.504839</td>
      <td>-0.135401</td>
      <td>-0.423513</td>
    </tr>
    <tr>
      <th>265434</th>
      <td>0.245442</td>
      <td>0.757524</td>
      <td>0.069692</td>
      <td>-0.675437</td>
      <td>0.834176</td>
      <td>-0.494112</td>
      <td>1.009999</td>
      <td>-0.188276</td>
      <td>-0.293707</td>
      <td>-0.637338</td>
      <td>...</td>
      <td>-0.212164</td>
      <td>-0.547161</td>
      <td>0.116291</td>
      <td>0.624563</td>
      <td>-0.456012</td>
      <td>0.097241</td>
      <td>-0.027043</td>
      <td>0.000804</td>
      <td>-0.135960</td>
      <td>0.906872</td>
    </tr>
    <tr>
      <th>92081</th>
      <td>-1.810611</td>
      <td>-6.514130</td>
      <td>-0.454912</td>
      <td>0.306467</td>
      <td>-3.931548</td>
      <td>-0.132263</td>
      <td>0.750604</td>
      <td>-0.541343</td>
      <td>-1.549850</td>
      <td>0.502068</td>
      <td>...</td>
      <td>1.010159</td>
      <td>-0.306694</td>
      <td>-1.448084</td>
      <td>1.042912</td>
      <td>-0.303816</td>
      <td>-0.260250</td>
      <td>-0.233200</td>
      <td>0.330292</td>
      <td>23.287221</td>
      <td>-0.245832</td>
    </tr>
  </tbody>
</table>
<p>284807 rows × 30 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-f926d18e-cce7-434a-8232-f81059bf2aad')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-f926d18e-cce7-434a-8232-f81059bf2aad button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-f926d18e-cce7-434a-8232-f81059bf2aad');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code execution_count="119" colab="{\"height\":762,\"base_uri\":\"https://localhost:8080/\"}" id="XN_Cd7tVpAga" outputId="c80a9c9d-774a-4903-a19c-be9b24d6f000"}
``` {.python}
# is not applied any oversampling method

# Create the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

# Fit a decision tree model to our data
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Obtain model predictions
y_predicted = model.predict(X_test)

# Calculate average precision 
average_precision = average_precision_score(y_test, y_predicted)

# Obtain precision and recall 
precision, recall, _ = precision_recall_curve(y_test, y_predicted)

# Print the classifcation report and confusion matrix
print('Classification report:\n', classification_report(y_test, y_predicted))

cm = confusion_matrix(y_test, y_predicted) # Support Vector Machine
plt.rcParams['figure.figsize'] = (8, 8)
svm_cm_plot = plot_confusion_matrix(cm, 
                                classes = ['0','1'], 
                                normalize = False, title = 'Confusion matrix of Decision tree')
plt.savefig('svm_cm_plot.png')
plt.show()
```

::: {.output .stream .stdout}
    Classification report:
                   precision    recall  f1-score   support

               0       1.00      1.00      1.00     56854
               1       0.75      0.73      0.74       108

        accuracy                           1.00     56962
       macro avg       0.87      0.87      0.87     56962
    weighted avg       1.00      1.00      1.00     56962
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/2ce4204064ea7a7c13c7da2912447fea05562fa2.png)
:::
:::

::: {.cell .code execution_count="120" colab="{\"height\":1000,\"base_uri\":\"https://localhost:8080/\"}" id="zhh6jHSpqJSM" outputId="c4817c41-8d69-43f7-dc28-1eb25555eeef"}
``` {.python}
# Plot the recall precision tradeoff
plot_pr_curve(recall, precision, average_precision)
plot_roc_curve(false_positive_rate,true_positive_rate,roc_auc)
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/92e75cfb49fa4ee987ab5eedc22a0e3f07b810d9.png)
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/ac61c11d31a5b9a4b6b56c8417102d3c1f9aa1f0.png)
:::
:::

::: {.cell .code execution_count="122" colab="{\"height\":1000,\"base_uri\":\"https://localhost:8080/\"}" id="qGJgnfBrqN5B" outputId="2d9859e6-3611-4985-86b1-835f49b7e088"}
``` {.python}
#comparison of two SMOTE oversampling techniques
#since the borderline SMOTE has the best value of recall, this would be used for training the model by using a pipeline

# Create the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=0)

# Define the resampling method
smote = SMOTE(random_state=0)
borderlinesmote = BorderlineSMOTE(kind='borderline-2',random_state=0)

# resample the training data
X_train_smote, y_train_smote = smote.fit_resample(X_train,y_train)
X_train_blsmote, y_train_blsmote = borderlinesmote.fit_resample(X_train,y_train)

# Fit a decision tree model to our data

smote_model = DecisionTreeClassifier().fit(X_train_smote, y_train_smote)
blsmote_model = DecisionTreeClassifier().fit(X_train_blsmote, y_train_blsmote)


y_smote = smote_model.predict(X_test)
y_blsmote = blsmote_model.predict(X_test)


print('Classifcation report:\n', classification_report(y_test, y_smote))
cm = confusion_matrix(y_test, y_smote) # Support Vector Machine
plt.rcParams['figure.figsize'] = (8, 8)
svm_cm_plot = plot_confusion_matrix(cm, 
                                classes = ['0','1'], 
                                normalize = False, title = 'Confusion matrix of Decision tree')
plt.savefig('svm_cm_plot.png')
plt.show()
print('*'*25)

print('Classifcation report:\n', classification_report(y_test, y_blsmote))
cm = confusion_matrix(y_test, y_blsmote) # Support Vector Machine
plt.rcParams['figure.figsize'] = (8, 8)
svm_cm_plot = plot_confusion_matrix(cm, 
                                classes = ['0','1'], 
                                normalize = False, title = 'Confusion matrix of Decision tree')
plt.savefig('svm_cm_plot.png')
plt.show()
print('*'*25)
```

::: {.output .stream .stdout}
    Classifcation report:
                   precision    recall  f1-score   support

               0       1.00      1.00      1.00     56854
               1       0.40      0.73      0.52       108

        accuracy                           1.00     56962
       macro avg       0.70      0.86      0.76     56962
    weighted avg       1.00      1.00      1.00     56962
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/946dc7c3a8876659d451866ea7df99587941733a.png)
:::

::: {.output .stream .stdout}
    *************************
    Classifcation report:
                   precision    recall  f1-score   support

               0       1.00      1.00      1.00     56854
               1       0.57      0.73      0.64       108

        accuracy                           1.00     56962
       macro avg       0.78      0.87      0.82     56962
    weighted avg       1.00      1.00      1.00     56962
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/75b2d3a34fb86d78b32a8c86daf6fb2c59785734.png)
:::

::: {.output .stream .stdout}
    *************************
:::
:::

::: {.cell .code execution_count="123" colab="{\"height\":762,\"base_uri\":\"https://localhost:8080/\"}" id="9SSKF5ltq_I4" outputId="1a8e2d03-8fab-4117-ff28-bd028960a828"}
``` {.python}
# Create the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=0)
# Define which resampling method and which ML model to use in the pipeline
resampling = BorderlineSMOTE()
model = DecisionTreeClassifier() 

# Define the pipeline, tell it to combine SMOTE with the Decision tree model
pipeline = Pipeline([('SMOTE', resampling), ('Decision Tree Classifier', model)])

# Fit your pipeline onto your training set and obtain predictions by fitting the model onto the test data 
pipeline.fit(X_train, y_train) 
y_predicted = pipeline.predict(X_test)

# Obtain the results from the classification report and confusion matrix 
print('Classifcation report:\n', classification_report(y_test, y_predicted))

cm = confusion_matrix(y_test, y_predicted) # Support Vector Machine
plt.rcParams['figure.figsize'] = (8, 8)
svm_cm_plot = plot_confusion_matrix(cm, 
                                classes = ['0','1'], 
                                normalize = False, title = 'Confusion matrix of Decision tree')
plt.savefig('svm_cm_plot.png')
plt.show()
```

::: {.output .stream .stdout}
    Classifcation report:
                   precision    recall  f1-score   support

               0       1.00      1.00      1.00     56854
               1       0.70      0.72      0.71       108

        accuracy                           1.00     56962
       macro avg       0.85      0.86      0.86     56962
    weighted avg       1.00      1.00      1.00     56962
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/44cabbdcb12928ea0291c787a3cd79e492a01890.png)
:::
:::

::: {.cell .code execution_count="124" colab="{\"height\":1000,\"base_uri\":\"https://localhost:8080/\"}" id="JFV4rTQnrlfJ" outputId="0851ed5f-57ea-4974-8f3d-4f9e285b1e84"}
``` {.python}
# Calculate average precision 
average_precision = average_precision_score(y_test, y_predicted)

# Obtain precision and recall 
precision, recall, _ = precision_recall_curve(y_test, y_predicted)

# Plot the roc curve 
plot_roc_curve(false_positive_rate,true_positive_rate,roc_auc)

# Plot recall precision curve
plot_pr_curve(recall, precision, average_precision)
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/ac61c11d31a5b9a4b6b56c8417102d3c1f9aa1f0.png)
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/4e5d9039d971cc8c6dba1b0e076eb14a0fedbcf4.png)
:::
:::

::: {.cell .markdown id="BQT2cBggnD0v"}
Training Part of Decision TREE after removing
\'V1\',\'V2\',\'V3\',\'V7\',\'V8\',\'V9\',\'V10\',\'V15\',\'V17\',\'V21\',\'V22\',\'V23\',\'V24\',\'V25\',\'V26\',\'V27\',\'V28\'
:::

::: {.cell .code execution_count="125" colab="{\"height\":270,\"base_uri\":\"https://localhost:8080/\"}" id="DWXQ85g7kTmC" outputId="1a0a251d-a662-4a8b-9e85-19c6eebdf29f"}
``` {.python}
# Scale "Time" and "Amount"
df['scaled_amount'] = RobustScaler().fit_transform(df['Amount'].values.reshape(-1,1))
#df['scaled_time'] = RobustScaler().fit_transform(df['Time'].values.reshape(-1,1))

# Make a new dataset named "df_scaled" dropping out original "Time" and "Amount"
df_scaled = df.drop(['Time','Amount','V1','V2','V3','V7','V8','V9','V10','V15','V17','V21','V22','V23','V24','V25','V26','V27','V28'],axis = 1,inplace=False)
df_scaled.head()
```

::: {.output .execute_result execution_count="125"}
```{=html}
  <div id="df-104ddb6b-c9fb-4049-aac2-61c6e5534e6f">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V11</th>
      <th>V12</th>
      <th>V13</th>
      <th>V14</th>
      <th>V16</th>
      <th>V18</th>
      <th>V19</th>
      <th>V20</th>
      <th>Class</th>
      <th>mins</th>
      <th>hours</th>
      <th>scaled_amount</th>
      <th>scaled_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>163467</th>
      <td>-0.739899</td>
      <td>1.076581</td>
      <td>-0.251537</td>
      <td>0.539103</td>
      <td>-0.138719</td>
      <td>-1.166615</td>
      <td>-0.608435</td>
      <td>0.628549</td>
      <td>0.448800</td>
      <td>0.133305</td>
      <td>-0.041285</td>
      <td>0</td>
      <td>12</td>
      <td>8</td>
      <td>-0.282401</td>
      <td>0.367345</td>
    </tr>
    <tr>
      <th>211403</th>
      <td>0.650530</td>
      <td>-0.058817</td>
      <td>-0.920016</td>
      <td>0.778779</td>
      <td>-0.075154</td>
      <td>-2.119390</td>
      <td>-0.606401</td>
      <td>0.067119</td>
      <td>0.587165</td>
      <td>0.142988</td>
      <td>-0.315038</td>
      <td>0</td>
      <td>26</td>
      <td>14</td>
      <td>-0.227206</td>
      <td>0.630928</td>
    </tr>
    <tr>
      <th>80125</th>
      <td>1.302911</td>
      <td>-1.550850</td>
      <td>0.900366</td>
      <td>0.377390</td>
      <td>1.326861</td>
      <td>-1.034335</td>
      <td>-0.217089</td>
      <td>-1.368498</td>
      <td>-0.560265</td>
      <td>1.398303</td>
      <td>0.309719</td>
      <td>0</td>
      <td>12</td>
      <td>16</td>
      <td>-0.228184</td>
      <td>-0.309520</td>
    </tr>
    <tr>
      <th>126826</th>
      <td>-2.280477</td>
      <td>-0.424261</td>
      <td>-1.015679</td>
      <td>-0.346110</td>
      <td>-0.435490</td>
      <td>0.866670</td>
      <td>-0.785313</td>
      <td>1.756455</td>
      <td>-1.530292</td>
      <td>-1.088577</td>
      <td>0.334034</td>
      <td>0</td>
      <td>41</td>
      <td>21</td>
      <td>1.453224</td>
      <td>-0.077257</td>
    </tr>
    <tr>
      <th>8221</th>
      <td>-0.513938</td>
      <td>-0.639434</td>
      <td>-0.543843</td>
      <td>2.576528</td>
      <td>-2.135597</td>
      <td>2.817757</td>
      <td>0.225021</td>
      <td>1.783841</td>
      <td>-0.159246</td>
      <td>0.534097</td>
      <td>0.159146</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>-0.084539</td>
      <td>-0.865118</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-104ddb6b-c9fb-4049-aac2-61c6e5534e6f')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-104ddb6b-c9fb-4049-aac2-61c6e5534e6f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-104ddb6b-c9fb-4049-aac2-61c6e5534e6f');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code execution_count="126" id="iL-7p_EQhQvn"}
``` {.python}
#Separating X and Y
X = df_scaled.drop(['Class', 'mins', 'hours','scaled_time'], axis=1)
y = df_scaled['Class']
```
:::

::: {.cell .code execution_count="127" colab="{\"height\":423,\"base_uri\":\"https://localhost:8080/\"}" id="RzuPD8_RlqOE" outputId="51501b51-a98f-457c-f271-d69c19c1460f"}
``` {.python}
X
```

::: {.output .execute_result execution_count="127"}
```{=html}
  <div id="df-3701be4d-e4c3-4186-8897-1abd340ccbcb">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V11</th>
      <th>V12</th>
      <th>V13</th>
      <th>V14</th>
      <th>V16</th>
      <th>V18</th>
      <th>V19</th>
      <th>V20</th>
      <th>scaled_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>163467</th>
      <td>-0.739899</td>
      <td>1.076581</td>
      <td>-0.251537</td>
      <td>0.539103</td>
      <td>-0.138719</td>
      <td>-1.166615</td>
      <td>-0.608435</td>
      <td>0.628549</td>
      <td>0.448800</td>
      <td>0.133305</td>
      <td>-0.041285</td>
      <td>-0.282401</td>
    </tr>
    <tr>
      <th>211403</th>
      <td>0.650530</td>
      <td>-0.058817</td>
      <td>-0.920016</td>
      <td>0.778779</td>
      <td>-0.075154</td>
      <td>-2.119390</td>
      <td>-0.606401</td>
      <td>0.067119</td>
      <td>0.587165</td>
      <td>0.142988</td>
      <td>-0.315038</td>
      <td>-0.227206</td>
    </tr>
    <tr>
      <th>80125</th>
      <td>1.302911</td>
      <td>-1.550850</td>
      <td>0.900366</td>
      <td>0.377390</td>
      <td>1.326861</td>
      <td>-1.034335</td>
      <td>-0.217089</td>
      <td>-1.368498</td>
      <td>-0.560265</td>
      <td>1.398303</td>
      <td>0.309719</td>
      <td>-0.228184</td>
    </tr>
    <tr>
      <th>126826</th>
      <td>-2.280477</td>
      <td>-0.424261</td>
      <td>-1.015679</td>
      <td>-0.346110</td>
      <td>-0.435490</td>
      <td>0.866670</td>
      <td>-0.785313</td>
      <td>1.756455</td>
      <td>-1.530292</td>
      <td>-1.088577</td>
      <td>0.334034</td>
      <td>1.453224</td>
    </tr>
    <tr>
      <th>8221</th>
      <td>-0.513938</td>
      <td>-0.639434</td>
      <td>-0.543843</td>
      <td>2.576528</td>
      <td>-2.135597</td>
      <td>2.817757</td>
      <td>0.225021</td>
      <td>1.783841</td>
      <td>-0.159246</td>
      <td>0.534097</td>
      <td>0.159146</td>
      <td>-0.084539</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9693</th>
      <td>0.271141</td>
      <td>-0.611746</td>
      <td>-0.355526</td>
      <td>0.922178</td>
      <td>-2.097269</td>
      <td>2.117776</td>
      <td>1.433868</td>
      <td>0.178439</td>
      <td>-0.336217</td>
      <td>-0.763692</td>
      <td>-0.092231</td>
      <td>-0.098512</td>
    </tr>
    <tr>
      <th>212424</th>
      <td>0.486926</td>
      <td>-0.992713</td>
      <td>0.013298</td>
      <td>0.288016</td>
      <td>0.237436</td>
      <td>-1.163642</td>
      <td>0.167894</td>
      <td>0.983904</td>
      <td>0.827513</td>
      <td>0.015058</td>
      <td>-0.134108</td>
      <td>0.669042</td>
    </tr>
    <tr>
      <th>59004</th>
      <td>1.349409</td>
      <td>2.587102</td>
      <td>-2.875825</td>
      <td>-0.280232</td>
      <td>0.340369</td>
      <td>-0.805294</td>
      <td>-0.926441</td>
      <td>-0.046123</td>
      <td>-0.241728</td>
      <td>-1.189369</td>
      <td>0.889643</td>
      <td>-0.135401</td>
    </tr>
    <tr>
      <th>265434</th>
      <td>-0.675437</td>
      <td>0.834176</td>
      <td>-0.494112</td>
      <td>-0.826358</td>
      <td>0.658776</td>
      <td>1.151147</td>
      <td>-0.099196</td>
      <td>-0.107955</td>
      <td>-0.891843</td>
      <td>-0.102322</td>
      <td>-0.070872</td>
      <td>-0.135960</td>
    </tr>
    <tr>
      <th>92081</th>
      <td>0.306467</td>
      <td>-3.931548</td>
      <td>-0.132263</td>
      <td>-0.101733</td>
      <td>0.070139</td>
      <td>1.839370</td>
      <td>-0.674524</td>
      <td>-0.307378</td>
      <td>0.351405</td>
      <td>-1.046309</td>
      <td>3.138489</td>
      <td>23.287221</td>
    </tr>
  </tbody>
</table>
<p>284807 rows × 12 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-3701be4d-e4c3-4186-8897-1abd340ccbcb')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-3701be4d-e4c3-4186-8897-1abd340ccbcb button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-3701be4d-e4c3-4186-8897-1abd340ccbcb');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code execution_count="130" colab="{\"height\":762,\"base_uri\":\"https://localhost:8080/\"}" id="iucwM1F-l_NQ" outputId="7c7fca3e-21fe-467d-d9f3-e42746606bb9"}
``` {.python}
# Create the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=0)
# Define which resampling method and which ML model to use in the pipeline
resampling = BorderlineSMOTE()
model = DecisionTreeClassifier(criterion="entropy", max_depth=4) 

# Define the pipeline, tell it to combine SMOTE with the Decision tree model
pipeline = Pipeline([('SMOTE', resampling), ('Decision Tree Classifier', model)])

# Fit your pipeline onto your training set and obtain predictions by fitting the model onto the test data 
pipeline.fit(X_train, y_train) 
y_predicted = pipeline.predict(X_test)

# Obtain the results from the classification report and confusion matrix 
print('Classifcation report:\n', classification_report(y_test, y_predicted))

cm = confusion_matrix(y_test, y_predicted) # Support Vector Machine
plt.rcParams['figure.figsize'] = (8, 8)
svm_cm_plot = plot_confusion_matrix(cm, 
                                classes = ['0','1'], 
                                normalize = False, title = 'Confusion matrix of Decision tree with pipeline')
plt.savefig('svm_cm_plot.png')
plt.show()
```

::: {.output .stream .stdout}
    Classifcation report:
                   precision    recall  f1-score   support

               0       1.00      0.98      0.99     56854
               1       0.07      0.82      0.13       108

        accuracy                           0.98     56962
       macro avg       0.53      0.90      0.56     56962
    weighted avg       1.00      0.98      0.99     56962
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/bd3d26dccb7726b593139058575bfa24f9b6ccf2.png)
:::
:::

::: {.cell .code execution_count="131" colab="{\"height\":1000,\"base_uri\":\"https://localhost:8080/\"}" id="TlSVdRx1mNYj" outputId="bbeca626-fb33-48f3-e5ab-dad07fea13a4"}
``` {.python}
# Calculate average precision 
average_precision = average_precision_score(y_test, y_predicted)

# Obtain precision and recall 
precision, recall, _ = precision_recall_curve(y_test, y_predicted)

# Plot the roc curve 
plot_roc_curve(false_positive_rate,true_positive_rate,roc_auc)

# Plot recall precision curve
plot_pr_curve(recall, precision, average_precision)
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/ac61c11d31a5b9a4b6b56c8417102d3c1f9aa1f0.png)
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/60d31953bf0124f34f6c6befc7d089eb11eefa53.png)
:::
:::

::: {.cell .code execution_count="134" colab="{\"height\":423,\"base_uri\":\"https://localhost:8080/\"}" id="appqrwtUwJqc" outputId="52a04a15-4017-48a1-b97a-f60c6fa98b49"}
``` {.python}
df1=X
df1
```

::: {.output .execute_result execution_count="134"}
```{=html}
  <div id="df-d9d7b04c-8b62-4250-aaa6-8b150511f058">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V11</th>
      <th>V12</th>
      <th>V13</th>
      <th>V14</th>
      <th>V16</th>
      <th>V18</th>
      <th>V19</th>
      <th>V20</th>
      <th>scaled_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>163467</th>
      <td>-0.739899</td>
      <td>1.076581</td>
      <td>-0.251537</td>
      <td>0.539103</td>
      <td>-0.138719</td>
      <td>-1.166615</td>
      <td>-0.608435</td>
      <td>0.628549</td>
      <td>0.448800</td>
      <td>0.133305</td>
      <td>-0.041285</td>
      <td>-0.282401</td>
    </tr>
    <tr>
      <th>211403</th>
      <td>0.650530</td>
      <td>-0.058817</td>
      <td>-0.920016</td>
      <td>0.778779</td>
      <td>-0.075154</td>
      <td>-2.119390</td>
      <td>-0.606401</td>
      <td>0.067119</td>
      <td>0.587165</td>
      <td>0.142988</td>
      <td>-0.315038</td>
      <td>-0.227206</td>
    </tr>
    <tr>
      <th>80125</th>
      <td>1.302911</td>
      <td>-1.550850</td>
      <td>0.900366</td>
      <td>0.377390</td>
      <td>1.326861</td>
      <td>-1.034335</td>
      <td>-0.217089</td>
      <td>-1.368498</td>
      <td>-0.560265</td>
      <td>1.398303</td>
      <td>0.309719</td>
      <td>-0.228184</td>
    </tr>
    <tr>
      <th>126826</th>
      <td>-2.280477</td>
      <td>-0.424261</td>
      <td>-1.015679</td>
      <td>-0.346110</td>
      <td>-0.435490</td>
      <td>0.866670</td>
      <td>-0.785313</td>
      <td>1.756455</td>
      <td>-1.530292</td>
      <td>-1.088577</td>
      <td>0.334034</td>
      <td>1.453224</td>
    </tr>
    <tr>
      <th>8221</th>
      <td>-0.513938</td>
      <td>-0.639434</td>
      <td>-0.543843</td>
      <td>2.576528</td>
      <td>-2.135597</td>
      <td>2.817757</td>
      <td>0.225021</td>
      <td>1.783841</td>
      <td>-0.159246</td>
      <td>0.534097</td>
      <td>0.159146</td>
      <td>-0.084539</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9693</th>
      <td>0.271141</td>
      <td>-0.611746</td>
      <td>-0.355526</td>
      <td>0.922178</td>
      <td>-2.097269</td>
      <td>2.117776</td>
      <td>1.433868</td>
      <td>0.178439</td>
      <td>-0.336217</td>
      <td>-0.763692</td>
      <td>-0.092231</td>
      <td>-0.098512</td>
    </tr>
    <tr>
      <th>212424</th>
      <td>0.486926</td>
      <td>-0.992713</td>
      <td>0.013298</td>
      <td>0.288016</td>
      <td>0.237436</td>
      <td>-1.163642</td>
      <td>0.167894</td>
      <td>0.983904</td>
      <td>0.827513</td>
      <td>0.015058</td>
      <td>-0.134108</td>
      <td>0.669042</td>
    </tr>
    <tr>
      <th>59004</th>
      <td>1.349409</td>
      <td>2.587102</td>
      <td>-2.875825</td>
      <td>-0.280232</td>
      <td>0.340369</td>
      <td>-0.805294</td>
      <td>-0.926441</td>
      <td>-0.046123</td>
      <td>-0.241728</td>
      <td>-1.189369</td>
      <td>0.889643</td>
      <td>-0.135401</td>
    </tr>
    <tr>
      <th>265434</th>
      <td>-0.675437</td>
      <td>0.834176</td>
      <td>-0.494112</td>
      <td>-0.826358</td>
      <td>0.658776</td>
      <td>1.151147</td>
      <td>-0.099196</td>
      <td>-0.107955</td>
      <td>-0.891843</td>
      <td>-0.102322</td>
      <td>-0.070872</td>
      <td>-0.135960</td>
    </tr>
    <tr>
      <th>92081</th>
      <td>0.306467</td>
      <td>-3.931548</td>
      <td>-0.132263</td>
      <td>-0.101733</td>
      <td>0.070139</td>
      <td>1.839370</td>
      <td>-0.674524</td>
      <td>-0.307378</td>
      <td>0.351405</td>
      <td>-1.046309</td>
      <td>3.138489</td>
      <td>23.287221</td>
    </tr>
  </tbody>
</table>
<p>284807 rows × 12 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d9d7b04c-8b62-4250-aaa6-8b150511f058')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-d9d7b04c-8b62-4250-aaa6-8b150511f058 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d9d7b04c-8b62-4250-aaa6-8b150511f058');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code execution_count="135" colab="{\"height\":1000,\"base_uri\":\"https://localhost:8080/\"}" id="mMwsXskQvxZe" outputId="3f7726c4-dc49-49b3-ef11-fc39d3e85a66"}
``` {.python}
plt.figure(figsize = (20,20))
dec_tree = plot_tree(decision_tree=model, feature_names = df1.columns, 
                   class_names =["0", "1"] , filled = True , precision = 4, fontsize=14)
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/45af8bd494ff47fe5f769c7af185e8ac06af7e01.png)
:::
:::

::: {.cell .markdown id="-EhgRQaPZL5u"}
**RandomForest Classifier**
:::

::: {.cell .code execution_count="136" colab="{\"height\":299,\"base_uri\":\"https://localhost:8080/\"}" id="IBrgOJRKNrTl" outputId="ab9d1f18-2a0c-4b38-d20d-9b6df2622969"}
``` {.python}
# Scale "Time" and "Amount"
df['scaled_amount'] = RobustScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = RobustScaler().fit_transform(df['Time'].values.reshape(-1,1))

# Make a new dataset named "df_scaled" dropping out original "Time" and "Amount"
df_scaled = df.drop(['Time','Amount'],axis = 1,inplace=False)
df_scaled.head()
```

::: {.output .execute_result execution_count="136"}
```{=html}
  <div id="df-2bfc1cde-75c3-47aa-b02f-b31a241536a9">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>...</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Class</th>
      <th>mins</th>
      <th>hours</th>
      <th>scaled_amount</th>
      <th>scaled_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>163467</th>
      <td>0.122040</td>
      <td>0.889577</td>
      <td>-0.557672</td>
      <td>-0.739899</td>
      <td>1.076581</td>
      <td>-0.251537</td>
      <td>0.768230</td>
      <td>0.150525</td>
      <td>-0.109897</td>
      <td>-0.638497</td>
      <td>...</td>
      <td>0.085934</td>
      <td>-0.457265</td>
      <td>0.126185</td>
      <td>0.214236</td>
      <td>0.067386</td>
      <td>0</td>
      <td>12</td>
      <td>8</td>
      <td>-0.282401</td>
      <td>0.367345</td>
    </tr>
    <tr>
      <th>211403</th>
      <td>1.983664</td>
      <td>-0.144566</td>
      <td>-1.601533</td>
      <td>0.650530</td>
      <td>-0.058817</td>
      <td>-0.920016</td>
      <td>-0.079494</td>
      <td>-0.052650</td>
      <td>0.791831</td>
      <td>-0.250719</td>
      <td>...</td>
      <td>-0.045930</td>
      <td>0.086611</td>
      <td>0.712647</td>
      <td>-0.066077</td>
      <td>-0.058296</td>
      <td>0</td>
      <td>26</td>
      <td>14</td>
      <td>-0.227206</td>
      <td>0.630928</td>
    </tr>
    <tr>
      <th>80125</th>
      <td>-3.095610</td>
      <td>2.341266</td>
      <td>0.978825</td>
      <td>1.302911</td>
      <td>-1.550850</td>
      <td>0.900366</td>
      <td>-1.023949</td>
      <td>1.546270</td>
      <td>1.244595</td>
      <td>1.157214</td>
      <td>...</td>
      <td>0.194447</td>
      <td>0.033323</td>
      <td>-0.509168</td>
      <td>-0.694157</td>
      <td>-0.489850</td>
      <td>0</td>
      <td>12</td>
      <td>16</td>
      <td>-0.228184</td>
      <td>-0.309520</td>
    </tr>
    <tr>
      <th>126826</th>
      <td>-0.677805</td>
      <td>-0.323111</td>
      <td>1.795775</td>
      <td>-2.280477</td>
      <td>-0.424261</td>
      <td>-1.015679</td>
      <td>0.842032</td>
      <td>-0.386281</td>
      <td>-1.098708</td>
      <td>-0.234947</td>
      <td>...</td>
      <td>0.365986</td>
      <td>-0.261502</td>
      <td>-0.817321</td>
      <td>-0.108201</td>
      <td>-0.071991</td>
      <td>0</td>
      <td>41</td>
      <td>21</td>
      <td>1.453224</td>
      <td>-0.077257</td>
    </tr>
    <tr>
      <th>8221</th>
      <td>1.365524</td>
      <td>-0.193916</td>
      <td>0.327665</td>
      <td>-0.513938</td>
      <td>-0.639434</td>
      <td>-0.543843</td>
      <td>-0.733840</td>
      <td>-0.151407</td>
      <td>0.173629</td>
      <td>-0.002654</td>
      <td>...</td>
      <td>-0.104268</td>
      <td>0.386347</td>
      <td>-0.321959</td>
      <td>0.002022</td>
      <td>0.025594</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>-0.084539</td>
      <td>-0.865118</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-2bfc1cde-75c3-47aa-b02f-b31a241536a9')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-2bfc1cde-75c3-47aa-b02f-b31a241536a9 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-2bfc1cde-75c3-47aa-b02f-b31a241536a9');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code execution_count="137" id="4n6H9TZlO64F"}
``` {.python}
#Separating X and Y
X = df_scaled.drop(['Class', 'mins', 'hours'], axis=1)
y = df_scaled['Class']
```
:::

::: {.cell .code execution_count="138" id="gxISz0oKO675"}
``` {.python}
#Train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.20, random_state = 0)
```
:::

::: {.cell .code execution_count="139" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="E31bqXlgO7C3" outputId="513ed7c8-9fc7-4215-9e11-9ccb4ae40a01"}
``` {.python}
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
classifier.predict_proba(X_test)
```

::: {.output .execute_result execution_count="139"}
    array([[1., 0.],
           [1., 0.],
           [1., 0.],
           ...,
           [1., 0.],
           [1., 0.],
           [1., 0.]])
:::
:::

::: {.cell .code execution_count="140" id="XiPbnOHLPDM7"}
``` {.python}
y_pred = classifier.predict(X_test)
```
:::

::: {.cell .code execution_count="141" colab="{\"height\":585,\"base_uri\":\"https://localhost:8080/\"}" id="4nxIqjpeU7W7" outputId="caebfc8b-38a9-4d90-bd0d-3c4dcc4e6df7"}
``` {.python}
# cm = confusion_matrix(y_test, y_predicted)
# plt.figure(figsize=(5,5))
# sns.heatmap(data=cm,linewidths=.5,annot=True,square=True,cmap='Blues')
# plt.ylabel('Actual label')
# plt.xlabel('Predicted label')
# all_sample_title = 'Accuracy Score: {0}'.format(classifier.score(X_test,y_test))
# plt.title(all_sample_title, size = 15)

cm = confusion_matrix(y_test, y_predicted) # Support Vector Machine
plt.rcParams['figure.figsize'] = (8, 8)
svm_cm_plot = plot_confusion_matrix(cm, 
                                classes = ['0','1'], 
                                normalize = False, title = 'Confusion matrix of Random Forest')
plt.savefig('svm_cm_plot.png')
plt.show()
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/b308b4f27e4c9f891e6f5c9de6ba5b687ea20471.png)
:::
:::

::: {.cell .code execution_count="142" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="kx4vmxDcVuCH" outputId="186b96f0-9579-4e77-8646-fa874ba4ac54"}
``` {.python}
print(classification_report(y_test, y_pred))
```

::: {.output .stream .stdout}
                  precision    recall  f1-score   support

               0       1.00      1.00      1.00     56854
               1       0.93      0.69      0.79       108

        accuracy                           1.00     56962
       macro avg       0.96      0.84      0.89     56962
    weighted avg       1.00      1.00      1.00     56962
:::
:::

::: {.cell .code execution_count="143" colab="{\"height\":762,\"base_uri\":\"https://localhost:8080/\"}" id="EEnf9_8uXthI" outputId="b9622658-3741-4ee4-ddfb-676177dd9d34"}
``` {.python}
# Create the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

# Define which resampling method and which ML model to use in the pipeline
resampling = BorderlineSMOTE(kind='borderline-2',random_state=0) # instead SMOTE(kind='borderline2') 
model =  RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0) 

# Define the pipeline, tell it to combine SMOTE with the Logistic Regression model
pipeline = Pipeline([('SMOTE', resampling), ('RandomForest Classifier', model)])

# Fit your pipeline onto your training set and obtain predictions by fitting the model onto the test data 
pipeline.fit(X_train, y_train) 
y_predicted = pipeline.predict(X_test)

# Obtain the results from the classification report and confusion matrix 
print('Classifcation report:\n', classification_report(y_test, y_predicted))

cm = confusion_matrix(y_test, y_predicted) # Support Vector Machine
plt.rcParams['figure.figsize'] = (8, 8)
svm_cm_plot = plot_confusion_matrix(cm, 
                                classes = ['0','1'], 
                                normalize = False, title = 'Confusion matrix of Random Forest')
plt.savefig('svm_cm_plot.png')
plt.show()
```

::: {.output .stream .stdout}
    Classifcation report:
                   precision    recall  f1-score   support

               0       1.00      1.00      1.00     56854
               1       0.82      0.78      0.80       108

        accuracy                           1.00     56962
       macro avg       0.91      0.89      0.90     56962
    weighted avg       1.00      1.00      1.00     56962
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/46e96cb863841f52a6b1a4400b65a1671a082b22.png)
:::
:::

::: {.cell .code execution_count="144" colab="{\"height\":1000,\"base_uri\":\"https://localhost:8080/\"}" id="5oPP4zNyYgMt" outputId="41d9c017-5d4d-49df-adf0-26db3c536e83"}
``` {.python}
# Calculate average precision 
average_precision = average_precision_score(y_test, y_predicted)

# Obtain precision and recall 
precision, recall, _ = precision_recall_curve(y_test, y_predicted)

# Plot the roc curve 
plot_roc_curve(false_positive_rate,true_positive_rate,roc_auc)

# Plot recall precision curve
plot_pr_curve(recall, precision, average_precision)
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/ac61c11d31a5b9a4b6b56c8417102d3c1f9aa1f0.png)
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/1695e01174ebbe056b6355b382510ade93fec06c.png)
:::
:::

::: {.cell .markdown id="gsfbqHzePhHr"}
Training Part of Random Forest after removing
\'V1\',\'V2\',\'V3\',\'V7\',\'V8\',\'V9\',\'V10\',\'V15\',\'V17\',\'V21\',\'V22\',\'V23\',\'V24\',\'V25\',\'V26\',\'V27\',\'V28\'
:::

::: {.cell .code execution_count="145" colab="{\"height\":270,\"base_uri\":\"https://localhost:8080/\"}" id="49dzs8QkPgFd" outputId="ed8a510b-c347-4a42-b2b2-3ffba0275a9c"}
``` {.python}
# Scale "Time" and "Amount"
df['scaled_amount'] = RobustScaler().fit_transform(df['Amount'].values.reshape(-1,1))
#df['scaled_time'] = RobustScaler().fit_transform(df['Time'].values.reshape(-1,1))

# Make a new dataset named "df_scaled" dropping out original "Time" and "Amount"
df_scaled = df.drop(['Time','Amount','V1','V2','V3','V7','V8','V9','V10','V15','V17','V21','V22','V23','V24','V25','V26','V27','V28'],axis = 1,inplace=False)
df_scaled.head()
```

::: {.output .execute_result execution_count="145"}
```{=html}
  <div id="df-ed350e7e-aaaa-4827-a8ec-7a0f3a4d2203">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V11</th>
      <th>V12</th>
      <th>V13</th>
      <th>V14</th>
      <th>V16</th>
      <th>V18</th>
      <th>V19</th>
      <th>V20</th>
      <th>Class</th>
      <th>mins</th>
      <th>hours</th>
      <th>scaled_amount</th>
      <th>scaled_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>163467</th>
      <td>-0.739899</td>
      <td>1.076581</td>
      <td>-0.251537</td>
      <td>0.539103</td>
      <td>-0.138719</td>
      <td>-1.166615</td>
      <td>-0.608435</td>
      <td>0.628549</td>
      <td>0.448800</td>
      <td>0.133305</td>
      <td>-0.041285</td>
      <td>0</td>
      <td>12</td>
      <td>8</td>
      <td>-0.282401</td>
      <td>0.367345</td>
    </tr>
    <tr>
      <th>211403</th>
      <td>0.650530</td>
      <td>-0.058817</td>
      <td>-0.920016</td>
      <td>0.778779</td>
      <td>-0.075154</td>
      <td>-2.119390</td>
      <td>-0.606401</td>
      <td>0.067119</td>
      <td>0.587165</td>
      <td>0.142988</td>
      <td>-0.315038</td>
      <td>0</td>
      <td>26</td>
      <td>14</td>
      <td>-0.227206</td>
      <td>0.630928</td>
    </tr>
    <tr>
      <th>80125</th>
      <td>1.302911</td>
      <td>-1.550850</td>
      <td>0.900366</td>
      <td>0.377390</td>
      <td>1.326861</td>
      <td>-1.034335</td>
      <td>-0.217089</td>
      <td>-1.368498</td>
      <td>-0.560265</td>
      <td>1.398303</td>
      <td>0.309719</td>
      <td>0</td>
      <td>12</td>
      <td>16</td>
      <td>-0.228184</td>
      <td>-0.309520</td>
    </tr>
    <tr>
      <th>126826</th>
      <td>-2.280477</td>
      <td>-0.424261</td>
      <td>-1.015679</td>
      <td>-0.346110</td>
      <td>-0.435490</td>
      <td>0.866670</td>
      <td>-0.785313</td>
      <td>1.756455</td>
      <td>-1.530292</td>
      <td>-1.088577</td>
      <td>0.334034</td>
      <td>0</td>
      <td>41</td>
      <td>21</td>
      <td>1.453224</td>
      <td>-0.077257</td>
    </tr>
    <tr>
      <th>8221</th>
      <td>-0.513938</td>
      <td>-0.639434</td>
      <td>-0.543843</td>
      <td>2.576528</td>
      <td>-2.135597</td>
      <td>2.817757</td>
      <td>0.225021</td>
      <td>1.783841</td>
      <td>-0.159246</td>
      <td>0.534097</td>
      <td>0.159146</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>-0.084539</td>
      <td>-0.865118</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ed350e7e-aaaa-4827-a8ec-7a0f3a4d2203')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-ed350e7e-aaaa-4827-a8ec-7a0f3a4d2203 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ed350e7e-aaaa-4827-a8ec-7a0f3a4d2203');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code execution_count="146" id="M-E-iUOmPt7t"}
``` {.python}
#Separating X and Y
X = df_scaled.drop(['Class', 'mins', 'hours'], axis=1)
y = df_scaled['Class']
```
:::

::: {.cell .code execution_count="147" colab="{\"height\":762,\"base_uri\":\"https://localhost:8080/\"}" id="vsUNzss5Pvad" outputId="7b21ffe3-00de-426c-f0ae-52a327583aeb"}
``` {.python}
# Create the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

# Define which resampling method and which ML model to use in the pipeline
resampling = BorderlineSMOTE(kind='borderline-2',random_state=0) # instead SMOTE(kind='borderline2') 
model =  RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0) 

# Define the pipeline, tell it to combine SMOTE with the Logistic Regression model
pipeline = Pipeline([('SMOTE', resampling), ('RandomForest Classifier', model)])

# Fit your pipeline onto your training set and obtain predictions by fitting the model onto the test data 
pipeline.fit(X_train, y_train) 
y_predicted = pipeline.predict(X_test)

# Obtain the results from the classification report and confusion matrix 
print('Classifcation report:\n', classification_report(y_test, y_predicted))

cm = confusion_matrix(y_test, y_predicted) # Support Vector Machine
plt.rcParams['figure.figsize'] = (8, 8)
svm_cm_plot = plot_confusion_matrix(cm, 
                                classes = ['0','1'], 
                                normalize = False, title = 'Confusion matrix of Random Forest')
plt.savefig('svm_cm_plot.png')
plt.show()
```

::: {.output .stream .stdout}
    Classifcation report:
                   precision    recall  f1-score   support

               0       1.00      1.00      1.00     56854
               1       0.82      0.77      0.79       108

        accuracy                           1.00     56962
       macro avg       0.91      0.88      0.90     56962
    weighted avg       1.00      1.00      1.00     56962
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/f2d2cf429de2eb4440d964d6460b469400f57ad8.png)
:::
:::

::: {.cell .code execution_count="148" colab="{\"height\":1000,\"base_uri\":\"https://localhost:8080/\"}" id="SbnpcR2UP69X" outputId="72e4f03e-72ac-4362-c411-1514849794bd"}
``` {.python}
# Calculate average precision 
average_precision = average_precision_score(y_test, y_predicted)

# Obtain precision and recall 
precision, recall, _ = precision_recall_curve(y_test, y_predicted)

# Plot the roc curve 
plot_roc_curve(false_positive_rate,true_positive_rate,roc_auc)

# Plot recall precision curve
plot_pr_curve(recall, precision, average_precision)
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/ac61c11d31a5b9a4b6b56c8417102d3c1f9aa1f0.png)
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/e29a3c66cd91ffd20afe11a94a7e35593c142c0f.png)
:::
:::

::: {.cell .markdown id="hls6uVYqc_VH"}
# (Support Vector Machine) SVM
:::

::: {.cell .code execution_count="153" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="GAAofP4vdD2g" outputId="5d59d8f9-6f7e-47bd-b6cd-592859a61f94"}
``` {.python}
cases = len(df)
nonfraud_count = len(df[df.Class == 0])
fraud_count = len(df[df.Class == 1])
fraud_percentage = round(fraud_count/nonfraud_count*100, 2)

print('Total number of cases are ',cases)
print('Number of Non-fraud cases are ',nonfraud_count)
print('Number of Fraud cases are ',fraud_count)
print('Percentage of fraud cases is {}',fraud_percentage)
```

::: {.output .stream .stdout}
    Total number of cases are  284807
    Number of Non-fraud cases are  284315
    Number of Fraud cases are  492
    Percentage of fraud cases is {} 0.17
:::
:::

::: {.cell .code execution_count="154" id="kS9QMoY7gjG4"}
``` {.python}
# DATA SPLIT
df['scaled_amount'] = RobustScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = RobustScaler().fit_transform(df['Time'].values.reshape(-1,1))

# Make a new dataset named "df_scaled" dropping out original "Time" and "Amount"
df_scaled = df.drop(['Time','Amount'],axis = 1,inplace=False)
df_scaled.head()

X = df_scaled.drop(['Class', 'mins', 'hours'], axis=1)
y = df_scaled['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```
:::

::: {.cell .code execution_count="155" id="U7uNzYmAgxKj"}
``` {.python}
svm = SVC()
svm.fit(X_train, y_train)
svm_ypred = svm.predict(X_test)

# Calculate average precision 
average_precision = average_precision_score(y_test, svm_ypred)

# Obtain precision and recall 
precision, recall, _ = precision_recall_curve(y_test, svm_ypred)
```
:::

::: {.cell .code execution_count="158" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="53ZQ2zWoheqX" outputId="4fd26776-df25-42a2-999a-12069945e55c"}
``` {.python}
print('Accuracy score of the SVM model is ',accuracy_score(y_test, svm_ypred))
```

::: {.output .stream .stdout}
    Accuracy score of the SVM model is  0.9992977774656788
:::
:::

::: {.cell .code execution_count="159" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="TxjuKa-OmF2m" outputId="3e3b1c9a-95b5-468d-8cc9-3848bb3de920"}
``` {.python}
print('F1 score of the SVM model is ',f1_score(y_test, svm_ypred))
```

::: {.output .stream .stdout}
    F1 score of the SVM model is  0.7938144329896906
:::
:::

::: {.cell .code execution_count="160" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="HfqbNQmWvO9H" outputId="d5acb4d2-ad46-4df7-b5b2-d0da077f17dc"}
``` {.python}
print('Classification report:\n', classification_report(y_test, svm_ypred))
```

::: {.output .stream .stdout}
    Classification report:
                   precision    recall  f1-score   support

               0       1.00      1.00      1.00     56854
               1       0.90      0.71      0.79       108

        accuracy                           1.00     56962
       macro avg       0.95      0.86      0.90     56962
    weighted avg       1.00      1.00      1.00     56962
:::
:::

::: {.cell .code execution_count="161" colab="{\"height\":585,\"base_uri\":\"https://localhost:8080/\"}" id="4FWtWR_KpeUD" outputId="c60ea9ab-98f3-4bbc-a814-5cdcc9ee975c"}
``` {.python}
cm = confusion_matrix(y_test, svm_ypred) # Support Vector Machine
plt.rcParams['figure.figsize'] = (8, 8)
svm_cm_plot = plot_confusion_matrix(cm, 
                                classes = ['0','1'], 
                                normalize = False, title = 'Confusion matrix of SVM')
plt.savefig('svm_cm_plot.png')
plt.show()
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/b1ca6ad7279526dd3fdf2e3c9ebf8ed2f2d23abc.png)
:::
:::

::: {.cell .code execution_count="162" colab="{\"height\":1000,\"base_uri\":\"https://localhost:8080/\"}" id="MSL57_fEspm0" outputId="f98ca702-1584-4489-c221-83e358d5c496"}
``` {.python}
# Plot the recall precision tradeoff
plot_pr_curve(recall, precision, average_precision)
plot_roc_curve(false_positive_rate,true_positive_rate,roc_auc)
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/4e380f32fea5831c0e0f2016756cc61fa57d5154.png)
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/ac61c11d31a5b9a4b6b56c8417102d3c1f9aa1f0.png)
:::
:::

::: {.cell .code execution_count="164" colab="{\"height\":780,\"base_uri\":\"https://localhost:8080/\"}" id="jntumyEIXOb8" outputId="3b02979e-95d7-4a08-d094-a700abaa0a25"}
``` {.python}
# Create the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define which resampling method and which ML model to use in the pipeline
resampling = SMOTE(random_state=0)
borderline_resampling = BorderlineSMOTE(kind='borderline-2',random_state=0) 
model = SVC() 

# Define the pipeline, tell it to combine SMOTE with the Logistic Regression model
borderline_pipeline = Pipeline([('SMOTE', borderline_resampling), ('SVM', model)])
pipeline = Pipeline([('SMOTE', resampling), ('SVM', model)])


# Fit your pipeline onto your training set and obtain predictions by fitting the model onto the test data 
borderline_pipeline.fit(X_train, y_train) 
svm_ypred_borderline = borderline_pipeline.predict(X_test)

print('Accuracy score of the SVM model is ', accuracy_score(y_test, svm_ypred_borderline))
print('Classifcation report with Borderline SMOTE:\n', classification_report(y_test, svm_ypred_borderline))

cm = confusion_matrix(y_test, svm_ypred_borderline) # Support Vector Machine
plt.rcParams['figure.figsize'] = (8, 8)
svm_cm_plot = plot_confusion_matrix(cm, 
                                classes = ['0','1'], 
                                normalize = False, title = 'Confusion matrix of SVM')
plt.savefig('svm_cm_plot.png')
plt.show()
```

::: {.output .stream .stdout}
    Accuracy score of the SVM model is  0.9945928864857273
    Classifcation report with Borderline SMOTE:
                   precision    recall  f1-score   support

               0       1.00      0.99      1.00     56854
               1       0.23      0.81      0.36       108

        accuracy                           0.99     56962
       macro avg       0.62      0.90      0.68     56962
    weighted avg       1.00      0.99      1.00     56962
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/3bc3673917be279344cec8cdaf7d150a69a6dbcd.png)
:::
:::

::: {.cell .code execution_count="165" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="qItACz_DwiF-" outputId="7f23ff27-fd56-43b4-b830-88e0f2ee68a0"}
``` {.python}
print('Accuracy score of the SVM model is ',accuracy_score(y_test, svm_ypred_borderline))
```

::: {.output .stream .stdout}
    Accuracy score of the SVM model is  0.9945928864857273
:::
:::

::: {.cell .code execution_count="166" colab="{\"height\":1000,\"base_uri\":\"https://localhost:8080/\"}" id="gRXmrTfKL5Ni" outputId="5a1e4dfa-c85c-43f0-8bce-69657dcce728"}
``` {.python}
# Calculate average precision 
average_precision = average_precision_score(y_test, svm_ypred_borderline)

# Obtain precision and recall 
precision, recall, _ = precision_recall_curve(y_test, svm_ypred_borderline)

# Plot the recall precision tradeoff
plot_pr_curve(recall, precision, average_precision)
plot_roc_curve(false_positive_rate,true_positive_rate,roc_auc)
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/64fe5c61f7637fa144808ec7ddb62d11259fa06e.png)
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/ac61c11d31a5b9a4b6b56c8417102d3c1f9aa1f0.png)
:::
:::

::: {.cell .markdown id="-cJpoj0xRPp8"}
Training Part of SVM after removing
\'V1\',\'V2\',\'V3\',\'V7\',\'V8\',\'V9\',\'V10\',\'V15\',\'V17\',\'V21\',\'V22\',\'V23\',\'V24\',\'V25\',\'V26\',\'V27\',\'V28\'
:::

::: {.cell .code execution_count="167" id="cW-HiNFAK2_B"}
``` {.python}
# Scale "Time" and "Amount"
from sklearn.preprocessing import StandardScaler, RobustScaler
df['scaled_amount'] = RobustScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = RobustScaler().fit_transform(df['Time'].values.reshape(-1,1))

# Make a new dataset named "df_scaled" dropping out original "Time" and "Amount"
df_scaled = df.drop(['Time','Amount','V1','V2','V3','V7','V8','V9','V10','V15','V17','V21','V22','V23','V24','V25','V26','V27','V28'],axis = 1,inplace=False)
df_scaled.head()

X = df_scaled.drop(['Class', 'mins', 'hours'], axis=1)
y = df_scaled['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```
:::

::: {.cell .code execution_count="168" colab="{\"height\":780,\"base_uri\":\"https://localhost:8080/\"}" id="f3xCMzkbn3dK" outputId="c65d5326-6c2c-46cb-a36f-42250f34cdfa"}
``` {.python}
# Create the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define which resampling method and which ML model to use in the pipeline
resampling = SMOTE(random_state=0)
borderline_resampling = BorderlineSMOTE(kind='borderline-2',random_state=0) # instead SMOTE(kind='borderline2') 
model = SVC() 

# Define the pipeline, tell it to combine SMOTE with the Logistic Regression model
borderline_pipeline = Pipeline([('SMOTE', borderline_resampling), ('SVM', model)])
pipeline = Pipeline([('SMOTE', resampling), ('SVM', model)])


# Fit your pipeline onto your training set and obtain predictions by fitting the model onto the test data 
borderline_pipeline.fit(X_train, y_train) 
svm_ypred_borderline = borderline_pipeline.predict(X_test)

# Obtain the results from the classification report and confusion matrix 

print('Accuracy score of the SVM model is ',accuracy_score(y_test, svm_ypred_borderline))
print('Classifcation report with Borderline SMOTE:\n', classification_report(y_test, svm_ypred_borderline))

cm = confusion_matrix(y_test, svm_ypred_borderline) # Support Vector Machine
plt.rcParams['figure.figsize'] = (8, 8)
svm_cm_plot = plot_confusion_matrix(cm, 
                                classes = ['0','1'], 
                                normalize = False, title = 'Confusion matrix of SVM with SMOTE')
plt.savefig('svm_cm_plot.png')
plt.show()
```

::: {.output .stream .stdout}
    Accuracy score of the SVM model is  0.9933991081773814
    Classifcation report with Borderline SMOTE:
                   precision    recall  f1-score   support

               0       1.00      0.99      1.00     56854
               1       0.19      0.79      0.31       108

        accuracy                           0.99     56962
       macro avg       0.60      0.89      0.65     56962
    weighted avg       1.00      0.99      1.00     56962
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/a9c05217a47c445e14a7c2aeece906d30e9cbca5.png)
:::
:::

::: {.cell .code execution_count="169" colab="{\"height\":1000,\"base_uri\":\"https://localhost:8080/\"}" id="tP9X08Hyn7Fy" outputId="c4fb18af-4c68-4cd4-daeb-2735e6099ebc"}
``` {.python}
# Calculate average precision 
average_precision = average_precision_score(y_test, svm_ypred_borderline)

# Obtain precision and recall 
precision, recall, _ = precision_recall_curve(y_test, svm_ypred_borderline)

# Plot the recall precision tradeoff
plot_pr_curve(recall, precision, average_precision)
plot_roc_curve(false_positive_rate,true_positive_rate,roc_auc)
```

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/eed7c97c1e94cca2a59e70ddab358cf2cf7fb7f6.png)
:::

::: {.output .display_data}
![](vertopal_b640231b5d3e43bcabdf79019a564a44/ac61c11d31a5b9a4b6b56c8417102d3c1f9aa1f0.png)
:::
:::
