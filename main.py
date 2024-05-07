import os
import pickle as pk
import pandas as pd
import streamlit
print("Current working directory:",os.getcwd())
list_features = [' Net Income to Total Assets',
                  ' Borrowing dependency',
                 ' Total income/Total expense',
                 ' Persistent EPS in the Last Four Seasons',
                 ' ROA(A) before interest and % after tax',
                 ' Retained Earnings to Total Assets',
                 ' Liability to Equity',
                 ' Total debt/Total net worth',
                 ' Equity to Liability',
                 ' Pre-tax net Interest Rate']
with open('model_select.pkl', 'rb') as model:
    model_use = pk.load(model)


# Creating model that perform classifications on user's inputs
def app_model(input_df):
    prediction = model_use.predict(input_df)
    return prediction


def wrangle(data):
    data[' Net Income to Total Assets'] = data['net_assets']
    data[' Borrowing_dependency'] = data['borrowing_dependency']
    data[' Total income/Total expense'] = data['expense']
    data[' Persistent EPS in the Last Four Seasons'] = data['seasons']
    data[' ROA(A) before interest and % after tax'] = data['tax']
    data[' Retained Earnings to Total Assets'] = data['assets']
    data[' Liability to Equity'] = data['equity']
    data[' Total debt/Total net worth'] = data['net_worth']
    data[' Equity to Liability'] = data['liability']
    data[' Pre-tax net Interest Rate'] = data['interest_rate']
    return data


def main():
    streamlit.title('TAIWAN COMPANIES BANKRUPTCY PREDICTIONS')
    streamlit.image('bankruptcy.jpeg')
    streamlit.write("""
        ## Fill the form below
    """)
    # accepting the important features form user
    net_assets = streamlit.number_input('Net Income to Total Assets', max_value=1, min_value=0)
    borrowing_dependency = streamlit.number_input('Borrowing_dependency', max_value=1, min_value=0)
    expense = streamlit.number_input('Total income/Total expense', max_value=1, min_value=0)
    seasons = streamlit.number_input('Persistent EPS in the Last Four Seasons', max_value=1, min_value=0)
    tax = streamlit.number_input('ROA(A) before interest and % after tax', max_value=1, min_value=0)
    earning = streamlit.number_input('Retained Earnings to Total Assets', max_value=1, min_value=0)
    equity = streamlit.number_input('Liability to Equity', max_value=1, min_value=0)
    net_worth = streamlit.number_input('Total debt/Total net worth', max_value=1, min_value=0)
    liability = streamlit.number_input('Equity to Liability', max_value=1, min_value=0)
    interest_rate = streamlit.number_input('Pre-tax net Interest Rate', max_value=1, min_value=0)

    if streamlit.button('PREDICT'):
        input_df = pd.DataFrame(
            {'net_Assets': [net_assets],
             'borrowing_dependency': [borrowing_dependency],
             'expense': [expense],
             'seasons': [seasons],
             'tax': [tax],
             'earning': [earning],
             'equity': [equity],
             'net_worth': [net_worth],
             'liability': [liability],
             'interest_Rate': [interest_rate]
             }
        )
        dataframe = wrangle(input_df)
        outcome = app_model(dataframe)
        if outcome == 1:
            streamlit.write("Negative")
        else:
            streamlit.write('Positive')


if __name__ == '__main__':
    main()
