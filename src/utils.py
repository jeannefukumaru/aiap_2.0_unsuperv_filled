import pandas as pd 
import numpy as np 

class Preprocess:

    def __init__(self):
        pass
                
    def clean_ecommerce(self, data):
        """
        input df : Kaggle ecommerce dataframe 
        output df: clean dataset with dropped duplicates, non-negative values
        and removed NAs, new columns, datetimes instead of strings etc.
        """
        data['CustomerID'] = data['CustomerID'].astype(str)  
        data['TotalPrice'] = data['Quantity']*data['UnitPrice']
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
        data.drop_duplicates()
        data = data[~(data['CustomerID']=='nan')]
        
        # remove decimals from CustomerID
        id_length = data['CustomerID'].str.len()[0]
        data['CustomerID'] = data['CustomerID'].str.slice(0,id_length-2)
        data = data[data['Quantity']>=0]
        return data

    def create_customer_data(self, data):
        """
        input df : clean Kaggle ecommerce dataframe
        output df: DataFrame with customers as rows
                    suitable for clustering
        """
        NoOfInvoices = data.groupby(['CustomerID'])['InvoiceNo'].nunique()
        NoOfUniqueItems = data.groupby(['CustomerID'])['StockCode'].nunique()
        QuantityPerInvoice = data.groupby(['CustomerID','InvoiceNo'])['Quantity'].sum()
        MeanQuantityPerInvoice = QuantityPerInvoice.groupby(['CustomerID']).mean()
        SpendingPerInvoice = data.groupby(['CustomerID','InvoiceNo'])['TotalPrice'].sum()
        MeanSpendingPerInvoice = SpendingPerInvoice.groupby(['CustomerID']).mean()
        UniqueItemsPerInvoice = data.groupby(['CustomerID', 'InvoiceNo']).nunique
        UnitPriceMean = data.groupby(['CustomerID'])['UnitPrice'].mean()
        UnitPriceStd = data.groupby(['CustomerID'])['UnitPrice'].std()
        TotalQuantity = data.groupby(['CustomerID'])['Quantity'].sum()
        customer = pd.DataFrame({'NoOfInvoices':NoOfInvoices, 'NoOfUniqueItems':NoOfUniqueItems, 
                            'MeanQuantityPerInvoice':MeanQuantityPerInvoice,'MeanSpendingPerInvoice':MeanSpendingPerInvoice,
                            'UniqueItemsPerInvoice':UniqueItemsPerInvoice,'UnitPriceMean':UnitPriceMean, 'UnitPriceStd':UnitPriceStd,
                            'TotalQuantity':TotalQuantity})
        return customer