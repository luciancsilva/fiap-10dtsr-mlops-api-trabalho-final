from pathlib import Path
import src.app as app
import json

def test_model_exists():
    arquivo_path = Path("model/model.pkl")
    assert arquivo_path.is_file(), "Model file does not exist at the specified path."

def test_model_version_exists():
    arquivo_path = Path("model/model_metadata.json")
    assert arquivo_path.is_file(), "Model version file does not exist at the specified path."

def test_handler_call():
    payload = {
        "Age": 30,
        "Annual_Income": 43391.96,
        "Num_Bank_Accounts": 1,
        "Num_Credit_Card": 5,
        "Num_of_Delayed_Payment": 6,
        "Credit_Utilization_Ratio": 29.112467941684688,
        "Payment_of_Min_Amount": 0,
        "Total_EMI_per_month": 0.0,
        "Credit_History_Age_Formated": 284,
        "Auto_Loan": 0,
        "Credit-Builder_Loan": 0,
        "Personal_Loan": 0,
        "Home_Equity_Loan": 0,
        "Mortgage_Loan": 0,
        "Student_Loan": 0,
        "Debt_Consolidation_Loan": 0,
        "Payday_Loan": 0,
        "Missed_Payment_Day": 1
    }

    event = {"data": payload}
    response = app.handler(event, None)

    response['body'] = json.loads(response['body'])

    assert isinstance(response["body"]["prediction"], float), "Prediction should be an integer"
    assert response["body"]["prediction"] > 0, "Prediction should be a non-negative integer"
    assert response["statusCode"] == 200, "Status code should be 200 OK"