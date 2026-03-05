from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import io

app = FastAPI()

@app.get("/")
def home():
    return {"message": "lokZI API running"}

@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)):

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    df.columns = df.columns.str.lower()

    quantity_col = None
    price_col = None
    date_col = None
    product_col = None

    for col in df.columns:
        if "quant" in col or "cant" in col:
            quantity_col = col
        if "price" in col or "precio" in col:
            price_col = col
        if "date" in col or "fecha" in col:
            date_col = col
        if "product" in col or "producto" in col:
            product_col = col

    if quantity_col and price_col:
        df["revenue"] = df[quantity_col] * df[price_col]
    else:
        return {"error": "No se detectaron columnas de precio y cantidad"}

    revenue_total = float(df["revenue"].sum())
    avg_ticket = float(df["revenue"].mean())

    if product_col:
        top_products = (
            df.groupby(product_col)["revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .to_dict()
        )
    else:
        top_products = {}

    forecast = []

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        monthly = df.groupby(pd.Grouper(key=date_col, freq="M"))["revenue"].sum().reset_index()

        monthly["t"] = np.arange(len(monthly))

        X = monthly[["t"]]
        y = monthly["revenue"]

        model = LinearRegression()
        model.fit(X, y)

        future = np.arange(len(monthly), len(monthly) + 3).reshape(-1, 1)
        preds = model.predict(future)

        forecast = preds.tolist()

    return {
        "revenue_total": revenue_total,
        "avg_ticket": avg_ticket,
        "top_products": top_products,
        "forecast_next_3_months": forecast
    }