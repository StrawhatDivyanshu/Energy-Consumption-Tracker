from fastapi import APIRouter
from fastapi.responses import JSONResponse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import io
import uvicorn

cost_api = APIRouter()

# Load the data file (adjust path as needed)
data = pd.read_csv(r'C:\Users\Asus\OneDrive\Desktop\Energy-tracker\Energy_Tracker\Energy-Consumption-Tracker\data\household_power_consumption_days.xls')
#data = pd.read_excel(data_file, engine='openpyxl') 

# Define constants for cost calculations
ELECTRICITY_RATE = 0.12  # Cost per unit of electricity (example)
GAS_RATE = 0.08          # Cost per unit of gas (example)

# Helper function to calculate costs
def calculate_monthly_cost(df):
    df['Total_Cost'] = df['Global_active_power'] * ELECTRICITY_RATE + df['Global_reactive_power'] * GAS_RATE
    return df.groupby(df['datetime'].dt.to_period('M')).agg({
        'Global_active_power': 'sum',
        'Global_reactive_power': 'sum',
        'Total_Cost': 'sum'
    }).reset_index()

@cost_api.get("/cost")
async def get_cost():
    # Parse data and ensure correct date format
    data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d')

    # Aggregate data by month
    monthly_data = calculate_monthly_cost(data)

    # Extract the last two months for comparison (if available)
    today = datetime.today()
    current_month = today.month
    last_month = current_month - 1 if current_month > 1 else 12
    current_year = today.year

    # Get data for this month and last month
    current_month_data = monthly_data[monthly_data['datetime'].dt.month == current_month]
    last_month_data = monthly_data[monthly_data['datetime'].dt.month == last_month]

    # Calculate cost summaries
    this_month_cost = current_month_data['Total_Cost'].sum()
    last_month_cost = last_month_data['Total_Cost'].sum()
    
    # Prepare monthly data for graph
    months = monthly_data['datetime'].dt.strftime('%b %Y').tolist()
    electricity_usage = monthly_data['Global_active_power'].tolist()
    gas_usage = monthly_data['Global_reactive_power'].tolist()

    # Generate monthly bar chart for electricity and gas usage
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = range(len(months))
    
    plt.bar(index, electricity_usage, bar_width, label='Electricity Usage', alpha=0.7)
    plt.bar([i + bar_width for i in index], gas_usage, bar_width, label='Gas Usage', alpha=0.7)
    
    plt.xlabel('Month')
    plt.ylabel('Usage')
    plt.title('Monthly Electricity and Gas Usage')
    plt.xticks([i + bar_width / 2 for i in index], months, rotation=45, ha='right')
    plt.legend()

    # Save the chart to a BytesIO object (in-memory file)
    chart_io = io.BytesIO()
    plt.tight_layout()
    plt.savefig(chart_io, format='png')
    chart_io.seek(0)

    # Prepare response data (cost summaries, chart image)
    response = {
        "thisMonth": round(this_month_cost, 2),
        "lastMonth": round(last_month_cost, 2),
        "monthlyChart": chart_io.getvalue().decode('latin1')  # Send image as base64 string
    }

    return JSONResponse(content=response)
if __name__ == "__main__":
    uvicorn.run(cost_api, host="0.0.0.0", port=8000)from fastapi import APIRouter
from fastapi.responses import JSONResponse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import io
import uvicorn

cost_api = APIRouter()

# Load the data file (adjust path as needed)
data = pd.read_csv(r'C:\Users\Asus\OneDrive\Desktop\Energy-tracker\Energy_Tracker\Energy-Consumption-Tracker\data\household_power_consumption_days.xls')
#data = pd.read_excel(data_file, engine='openpyxl') 

# Define constants for cost calculations
ELECTRICITY_RATE = 0.12  # Cost per unit of electricity (example)
GAS_RATE = 0.08          # Cost per unit of gas (example)

# Helper function to calculate costs
def calculate_monthly_cost(df):
    df['Total_Cost'] = df['Global_active_power'] * ELECTRICITY_RATE + df['Global_reactive_power'] * GAS_RATE
    return df.groupby(df['datetime'].dt.to_period('M')).agg({
        'Global_active_power': 'sum',
        'Global_reactive_power': 'sum',
        'Total_Cost': 'sum'
    }).reset_index()

@cost_api.get("/cost")
async def get_cost():
    # Parse data and ensure correct date format
    data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d')

    # Aggregate data by month
    monthly_data = calculate_monthly_cost(data)

    # Extract the last two months for comparison (if available)
    today = datetime.today()
    current_month = today.month
    last_month = current_month - 1 if current_month > 1 else 12
    current_year = today.year

    # Get data for this month and last month
    current_month_data = monthly_data[monthly_data['datetime'].dt.month == current_month]
    last_month_data = monthly_data[monthly_data['datetime'].dt.month == last_month]

    # Calculate cost summaries
    this_month_cost = current_month_data['Total_Cost'].sum()
    last_month_cost = last_month_data['Total_Cost'].sum()
    
    # Prepare monthly data for graph
    months = monthly_data['datetime'].dt.strftime('%b %Y').tolist()
    electricity_usage = monthly_data['Global_active_power'].tolist()
    gas_usage = monthly_data['Global_reactive_power'].tolist()

    # Generate monthly bar chart for electricity and gas usage
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = range(len(months))
    
    plt.bar(index, electricity_usage, bar_width, label='Electricity Usage', alpha=0.7)
    plt.bar([i + bar_width for i in index], gas_usage, bar_width, label='Gas Usage', alpha=0.7)
    
    plt.xlabel('Month')
    plt.ylabel('Usage')
    plt.title('Monthly Electricity and Gas Usage')
    plt.xticks([i + bar_width / 2 for i in index], months, rotation=45, ha='right')
    plt.legend()

    # Save the chart to a BytesIO object (in-memory file)
    chart_io = io.BytesIO()
    plt.tight_layout()
    plt.savefig(chart_io, format='png')
    chart_io.seek(0)

    # Prepare response data (cost summaries, chart image)
    response = {
        "thisMonth": round(this_month_cost, 2),
        "lastMonth": round(last_month_cost, 2),
        "monthlyChart": chart_io.getvalue().decode('latin1')  # Send image as base64 string
    }

    return JSONResponse(content=response)
if __name__ == "__main__":
    uvicorn.run(cost_api, host="0.0.0.0", port=8000)