# AI-Powered Market Intelligence System

A comprehensive end-to-end AI system that ingests data from multiple sources, performs intelligent analysis, and generates actionable market insights for mobile app and D2C e-commerce markets.

## 🎯 Project Overview

This system demonstrates modern AI/ML engineering practices by building a complete pipeline that:

- **Ingests & Cleans Data**: Processes messy CSV data and integrates live API feeds
- **Unifies Schemas**: Creates consistent data structures across different sources
- **Generates AI Insights**: Uses Google's Gemini AI to analyze user behavior and market trends
- **Interactive UI**: Streamlit-powered dashboard for exploring insights
- **Automated Reporting**: Generates executive summaries and actionable recommendations
- **Extensible Architecture**: Easily adaptable to different business verticals

## 🏗️ Architecture

```
├── data/                  # Data storage
│   ├── raw/              # Original datasets
│   └── processed/        # Cleaned and unified data
├── src/                  # Source code
│   ├── data_pipeline/    # Data loading and cleaning
│   ├── api_integration/  # External API clients
│   ├── analysis/         # AI-powered insights generation
│   ├── reporting/        # Automated report generation
│   └── app.py           # Main Streamlit application
└── notebooks/           # Jupyter notebooks for exploration
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- API Keys (included in .env)
  - RapidAPI Key for App Store data
  - Google Gemini API Key for AI insights

### Installation

1. **Clone and Setup**
   ```bash
   cd ai-market-intelligence
   pip install -r requirements.txt
   ```

2. **Prepare Data**
   - Place `googleplaystore.csv` in `data/raw/`
   - Place `d2c_dataset.csv` in `data/raw/`

3. **Run the Application**
   ```bash
   streamlit run src/app.py
   ```

### Docker Deployment

```bash
docker build -t ai-market-intelligence .
docker run -p 8501:8501 ai-market-intelligence
```

## 📊 Features

### Phase 1: Google Play Store Analysis
- **Data Cleaning**: Handles missing ratings, standardizes installs/size/price formats
- **Deduplication**: Intelligent duplicate removal based on app names and review counts
- **Data Validation**: Ensures proper data types and consistency

### Phase 2: Live API Integration
- **App Store Reviews**: Real-time review data via RapidAPI
- **Error Handling**: Robust retry mechanisms with exponential backoff
- **Schema Unification**: Combines Google Play and App Store data seamlessly

### Phase 3: AI-Powered Insights
- **Market Analysis**: Uses Gemini 2.5 Flash Lite for intelligent analysis
- **Feature Requests**: Identifies top user-requested features from reviews
- **Pain Points**: Discovers common complaints and issues
- **Confidence Scoring**: AI provides confidence levels for each insight
- **Structured Output**: JSON-formatted insights for easy integration

### Phase 4: Interactive Dashboard
- **Category Selection**: Filter insights by app category
- **Visual Analytics**: Charts and graphs powered by Plotly
- **Real-time Updates**: Dynamic content based on user selections

### Phase 5: D2C E-commerce Extension
- **Campaign Analytics**: CAC and ROAS calculations
- **Creative Generation**: AI-powered ad headlines and meta descriptions
- **SEO Optimization**: Automated content for better search visibility

## 🔧 Configuration

### Environment Variables
```
RAPIDAPI_KEY=your_rapidapi_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### Model Configuration
- **Primary AI Model**: `gemini-2.5-flash-lite`
- **Input Token Limit**: 1,048,576 tokens
- **Output Token Limit**: 65,536 tokens
- **Supported Inputs**: Text, image, video, audio, PDF

## 📈 Data Sources

1. **Static Dataset**: Kaggle Google Play Store dataset
2. **Live API**: RapidAPI App Store reviews
3. **D2C Dataset**: Synthetic e-commerce campaign data

## 🛠️ Development

### Adding New Data Sources
1. Create a new client in `src/api_integration/`
2. Add data cleaning functions in `src/data_pipeline/cleaner.py`
3. Update the unified schema as needed

### Extending AI Analysis
1. Create new prompt templates in `src/analysis/insights.py`
2. Add corresponding UI components in `src/app.py`
3. Update report generation in `src/reporting/generator.py`

## 📋 Project Status

- [x] **Phase 0**: Project scaffolding and setup
- [ ] **Phase 1**: Google Play Store data pipeline
- [ ] **Phase 2**: API integration and data unification
- [ ] **Phase 3**: AI-powered insight generation
- [ ] **Phase 4**: Interactive UI and reporting
- [ ] **Phase 5**: D2C e-commerce extension

## 🤝 Contributing

This project follows industry best practices:
- Clean, modular code architecture
- Comprehensive error handling
- Detailed documentation
- Professional git workflow

## 📄 License

Built for educational and professional demonstration purposes.

---

**Built by**: Next-Gen AI Marketing Team  
**Stack**: Python, Streamlit, Pandas, Google Gemini AI  
**Deployment**: Docker, Hugging Face Spaces Ready
