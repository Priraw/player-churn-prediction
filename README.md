# Player Churn Prediction

🎮 AI-powered retention analytics for gaming companies

## 🎯 Problem
Gaming companies lose $100M+ annually to player churn...

## 💡 Solution
End-to-end ML system predicting 7-day and 30-day churn with 92%+ accuracy


## 📈 Results
- ROC-AUC: 0.93 (7-day), 0.90 (30-day)
- Identifies 87% of churners with only 12% false positives
- Estimated $3.5M annual savings per 5,000 players

## 🛠️ Tech Stack
Python • XGBoost • FastAPI • React • Docker

## 🚀 Quick Start
# Clone repository
git clone https://github.com/YOUR-USERNAME/player-churn-prediction.git
cd player-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run data pipeline
python generate_data.py
python feature_engineering.py
python train_models.py

# Start API
python main.py

# Access API docs
open http://localhost:8000/docs



## 🔬 Notebooks

Explore the analysis:
1. [01_EDA.ipynb](notebooks/01_EDA.ipynb) - Exploratory Data Analysis
2. [02_Feature_Analysis.ipynb](notebooks/02_Feature_Analysis.ipynb) - Feature Importance
3. [03_Model_Evaluation.ipynb](notebooks/03_Model_Evaluation.ipynb) - Performance Analysis

## 📸 Screenshots
[Dashboard]

## 🎓 Key Learnings
### Technical
- Feature engineering > model complexity
- Production concerns drive architecture
- Monitoring is not optional

### Business
- Focus on actionable insights
- ROI matters more than accuracy
- Stakeholder communication is key

## 📖 Read More

**Medium Article**: [Building a Production-Ready Churn Prediction System](YOUR-MEDIUM-LINK)

Detailed writeup covering:
- Problem framing & data generation
- Feature engineering strategies
- Model selection & evaluation
- API design & deployment
- Business impact analysis

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 👤 Author

**Priyanka Rawat**

- Portfolio: [priyanka-rawat.com](https://www.priyanka-rawat.com/)
- LinkedIn: [linkedin.com/in/priyanka--rawat](https://www.linkedin.com/in/priyanka--rawat/)
- Medium: [@pri00raw](https://medium.com/@pri00raw)
- Email: pri00raw@gmail.com

## 🌟 Acknowledgments

- Dataset inspired by gaming industry research
- Thanks to Anthropic Claude for development assistance

---

**If you found this project helpful, please ⭐ star the repository!**
