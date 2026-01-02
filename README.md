# ⚡ Powercast-AI: Load Forecasting & Decision Support

An AI-based electrical load forecasting and decision support application designed for power system planning and control-room style visualization.

## 🌟 Key Features

- **Signal Preprocessing**: Savitzky–Golay smoothing (window=11, poly=2) to preserve trends while removing noise.
- **AI-Powered Forecasting**: Multi-horizon load forecasting using advanced Neural Networks (MLP/LSTM).
- **Decision Support**: 
    - **Unit Commitment**: Automated ON/OFF recommendations for generator units based on predicted demand + 10% spinning reserve.
    - **Maintenance Planning**: Identification of low-load windows for optimal maintenance scheduling.
- **Interactive Visualization**: Real-time plots of historical data, smoothed trends, and future forecasts.

## 🚀 Quick Start

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/YOUR_USERNAME/Powercast-AI.git
cd Powercast-AI
pip install -r requirements.txt
```

### 2. Run the App
Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

## 📂 Project Structure

- `app.py`: Main Streamlit dashboard.
- `core/`: Core logic modules.
  - `preprocessor.py`: Signal smoothing and normalization.
  - `model.py`: AI forecasting engine (scikit-learn fallback for compatibility).
  - `decision_support.py`: Rule-based generation scheduling logic.
- `utils/`: Utility scripts (e.g., sample data generator).
- `data/`: Folder for historical load datasets.

## 🛠️ Usage
1. **Upload Data**: Use the sidebar to upload a CSV file with `timestamp` and `load` columns.
2. **Set Parameters**: Adjust the look-back window and forecast horizon.
3. **Configure Generators**: Input generator capacities to see real-time unit recommendations.
4. **Analyze**: View predictions and maintenance suggestions on the interactive dashboard.

## 📄 License
MIT License
