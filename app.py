from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
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


def filter_data_by_range(df, time_range="month", start_date=None, end_date=None):
    """
    Filters the DataFrame based on the time range.
    - 'today': Filters data for the latest entry of the available date in the dataset.
    - 'month': Filters data from the last 30 days relative to the most recent date in the dataset.
    - 'year': Filters data for the current and previous year based on the most recent date in the dataset.
    - Custom range using start_date and end_date.
    """
    # Ensure 'date' column is in datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Get the most recent date in the dataset
    latest_date_in_data = df['datetime'].max()
    
    if time_range == "today":
        # Filter for the latest data point for the available day
        df_filtered = df[df['datetime'] == latest_date_in_data]

    elif time_range == "month":
        # Get data from the last 30 days relative to the most recent date in the dataset
        thirty_days_ago = latest_date_in_data - timedelta(days=30)
        df_filtered = df[df['datetime'] >= thirty_days_ago]

    elif time_range == "year":
        # Get data for the current and previous year relative to the most recent date in the dataset
        current_year = latest_date_in_data.year
        start_of_current_year = datetime(current_year, 1, 1)
        start_of_previous_year = datetime(current_year - 1, 1, 1)
        
        # Filter data for current and previous year
        df_filtered = df[df['datetime'] >= start_of_previous_year]

    elif start_date and end_date:
        # Custom date range filtering
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df_filtered = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
    
    else:
        # For unknown time ranges, return the full DataFrame
        df_filtered = df

    return df_filtered

# Route for dashboard metrics
@app.get("/dashboard")
async def dashboard_metrics(time_range: str = Query("today", description="Time range: today, month, year")):
    try:
        # Get the latest date from the dataset (assuming the column is 'timestamp' and it's in datetime format)
        latest_date = data['datetime'].max()
        
        # Use the latest date from the dataset
        current_date = latest_date

        # Get the start of the day, month, and year based on the latest date in the dataset
        current_day_start = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
        current_month_start = current_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # Determine the start and end date based on time_range
        if time_range == "today":
            # For 'today', filter only the latest data point in the day
            start_date = current_day_start
            end_date = current_date
        elif time_range == "month":
            # For 'month', filter from the start of the month to today
            start_date = current_month_start
            end_date = current_date
        elif time_range == "year":
            # For 'year', filter from the start of the year to today
            start_date = current_date.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date = current_date
        else:
            return {"error": "Invalid time range"}

        # Filter data based on the time range
        filtered_data = filter_data_by_range(data, time_range, start_date=start_date, end_date=end_date)

        # Calculate metrics for the current period
        total_power = filtered_data['Global_active_power'].sum()
        costs = {
            "cost_0.10": total_power * 0.10,
            "cost_0.15": total_power * 0.15,
            "cost_0.20": total_power * 0.20,
            "cost_0.30": total_power * 0.30
        }

        # Calculate change in cost based on the time range
        if time_range == "month":
            # Filter previous month's data
            previous_month_start = (current_month_start - timedelta(days=1)).replace(day=1)
            previous_month_end = current_month_start - timedelta(days=1)
            previous_month_data = filter_data_by_range(data, "month", start_date=previous_month_start, end_date=previous_month_end)
            previous_total_power = previous_month_data['Global_active_power'].sum() if not previous_month_data.empty else 0
            
            change_in_cost = {
                "previous_cost": {
                    "time_range": f"month {previous_month_start.strftime('%B %Y')}",
                    "cost_0.10": previous_total_power * 0.10,
                    "cost_0.15": previous_total_power * 0.15,
                    "cost_0.20": previous_total_power * 0.20,
                    "cost_0.30": previous_total_power * 0.30
                },
                "current_cost": {
                    "time_range": f"month {current_date.strftime('%B %Y')}",
                    "cost_0.10": total_power * 0.10,
                    "cost_0.15": total_power * 0.15,
                    "cost_0.20": total_power * 0.20,
                    "cost_0.30": total_power * 0.30
                },
                "percentage_change": ((total_power - previous_total_power) / previous_total_power * 100) if previous_total_power > 0 else None
            }
        elif time_range == "year":
            current_year = current_date.year
            previous_year = current_year - 1

            # Calculate the start and end dates for the current and previous year
            current_year_start = datetime(current_year, 1, 1)
            current_year_end = datetime(current_year, 12, 31, 23, 59, 59)
            previous_year_start = datetime(previous_year, 1, 1)
            previous_year_end = datetime(previous_year, 12, 31, 23, 59, 59)

            # Filter data for the current and previous year using start and end dates
            current_year_data = filter_data_by_range(data, "year", start_date=current_year_start, end_date=current_year_end)
            previous_year_data = filter_data_by_range(data, "year", start_date=previous_year_start, end_date=previous_year_end)

            previous_total_power = previous_year_data['Global_active_power'].sum() if not previous_year_data.empty else 0

            change_in_cost = {
                "previous_cost": {
                    "time_range": f"year {previous_year}",
                    "cost_0.10": previous_total_power * 0.10,
                    "cost_0.15": previous_total_power * 0.15,
                    "cost_0.20": previous_total_power * 0.20,
                    "cost_0.30": previous_total_power * 0.30
                },
                "current_cost": {
                    "time_range": f"year {current_year}",
                    "cost_0.10": total_power * 0.10,
                    "cost_0.15": total_power * 0.15,
                    "cost_0.20": total_power * 0.20,
                    "cost_0.30": total_power * 0.30
                },
                "percentage_change": ((total_power - previous_total_power) / previous_total_power * 100) if previous_total_power > 0 else None
            }

        else:
            # For 'today', there's no previous cost, so return the current cost
            change_in_cost = {
                "current_cost": {
                    "time_range": f"today {current_date.strftime('%Y-%m-%d')}",
                    "cost_0.10": total_power * 0.10,
                    "cost_0.15": total_power * 0.15,
                    "cost_0.20": total_power * 0.20,
                    "cost_0.30": total_power * 0.30
                }
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

        # Return the response including the date range and costs
        return {
            "date_range": {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "time_range": time_range
            },
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