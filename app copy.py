from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import uvicorn
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from typing import Optional

# Load the trained ARIMA model
with open('model.pkl', 'rb') as model_file:
    arima_model = pickle.load(model_file)

# Load the updated daily dataset
data = pd.read_csv('./data/household_power_consumption_days.xls')

# Parse datetime column for filtering purposes
data['datetime'] = pd.to_datetime(data['datetime'])

# Initialize FastAPI app
app = FastAPI()

# Define input schema for prediction
class PredictionInput(BaseModel):
    date: str  # Date in format yyyy-mm-dd

# Utility function to filter data by time range
# Utility function to filter data by time range
def filter_data_by_range(data, time_range):
    print(f"Time Range: {time_range}")
    if time_range == "today":
        filtered_data = data[data['datetime'] == data['datetime'].max()]
    elif time_range == "month":
        filtered_data = data[data['datetime'] >= data['datetime'].max() - pd.Timedelta(days=30)]
    elif time_range == "year":
        filtered_data = data[data['datetime'] >= data['datetime'].max() - pd.Timedelta(days=365)]
    else:
        # Raise an HTTP exception if the time_range is invalid
        raise HTTPException(status_code=400, detail="Invalid time range. Use 'today', 'month', or 'year'.")
    return filtered_data

# Route for dashboard metrics
@app.get("/dashboard")
async def dashboard_metrics(time_range: str = Query("today", description="Time range: today, month, year")):
    try:
        print(f"Time Range: {time_range}")  # Add this line to debug the value

        # Filter data based on time range
        filtered_data = filter_data_by_range(data, time_range)

        # Calculate metrics
        total_power = filtered_data['Global_active_power'].sum()
        costs = {
            "cost_0.10": total_power * 0.10,
            "cost_0.15": total_power * 0.15,
            "cost_0.20": total_power * 0.20,
            "cost_0.30": total_power * 0.30
        }

        # Calculate change in cost
        # Calculate change in cost (use correct previous time range)
        previous_data = filter_data_by_range(data, 
            "month" if time_range == "today" else 
            "year" if time_range in ["month", "year"] else "")
        previous_total_power = previous_data['Global_active_power'].sum() if not previous_data.empty else 0

        change_in_cost = {
            "previous_cost": previous_total_power * 0.10,  # Example using cost_0.10
            "current_cost": total_power * 0.10,
            "percentage_change": ((total_power - previous_total_power) / previous_total_power * 100) if previous_total_power > 0 else None
        }

        # Calculate usage estimate (current and predicted)
        current_usage = total_power
        predicted_usage = arima_model.forecast(steps=1)[0]  # Example: one-step forecast

        # Calculate energy intensity
        energy_intensity = filtered_data['Global_intensity'].sum() / total_power if total_power > 0 else None

        # Calculate active appliances
        active_appliances = {
            "sub_metering_1": filtered_data['Sub_metering_1'].sum(),
            "sub_metering_2": filtered_data['Sub_metering_2'].sum(),
            "sub_metering_3": filtered_data['Sub_metering_3'].sum(),
            "sub_metering_4": filtered_data['sub_metering_4'].sum()
        }

        return {
            "costs": costs,
            "change_in_cost": change_in_cost,
            "usage_estimate": {
                "current": current_usage,
                "predicted": predicted_usage
            },
            "energy_intensity": energy_intensity,
            "active_appliances": active_appliances
        }
    except Exception as e:
        return {"error": str(e)}

# Route for predictions
@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Convert input date to datetime
        date_input = pd.to_datetime(input_data.date)

        # Make prediction (dummy example; replace with actual logic)
        prediction = arima_model.forecast(steps=1)[0]  # Example: one-step forecast

        return {
            "date": input_data.date,
            "predicted_global_active_power": prediction
        }
    except Exception as e:
        return {"error": str(e)}

# Route to fetch data
@app.get("/data")
async def get_data(start_date: str, end_date: str):
    try:
        # Convert input dates to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Filter data by date range
        filtered_data = data[(data['datetime'] >= start_date) & (data['datetime'] <= end_date)]
        return filtered_data.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

# Route for chart generation
@app.get("/chart")
async def generate_chart(column: str = Query(..., description="Column to visualize"), time_range: str = Query("today", description="Time range: today, month, year")):
    try:
        # Filter data by time range
        filtered_data = filter_data_by_range(data, time_range)

        # Create a plot for the specified column
        plt.figure(figsize=(10, 5))
        plt.plot(filtered_data['datetime'], filtered_data[column], label=column)
        plt.title(f"Visualization for {column} ({time_range})")
        plt.xlabel("Date")
        plt.ylabel(column)
        plt.legend()

        # Save the chart as an image
        chart_path = "chart.png"
        plt.savefig(chart_path)
        plt.close()

        return {"chart_path": chart_path}
    except Exception as e:
        return {"error": str(e)}

# Run the app using the command: uvicorn app:app --reload
# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)