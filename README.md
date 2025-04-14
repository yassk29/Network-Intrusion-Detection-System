# 🚨 Network Packet Intrusion Detection System (NIDS)

A packet analysis and intrusion detection system with a sleek Streamlit dashboard interface.

---

## 📦 Features

- Fast model (lightweight)
- Slow model (heavy, accurate)
- Dual-model decision system with confidence threshold
- Real-time metrics (latency, packets/sec, F1, accuracy, etc.)
- Visual dashboard for model comparison & confusion matrix

---

## 🔧 Version 1 Capabilities

✅ Packet classification using test CSV  
✅ Streamlit dashboard with 3-model comparison  
✅ Metrics visualization + confusion matrix  

---

## 📂 Files

- `app.py`: Streamlit dashboard
- `fast.py`, `slow.py`, `final6.py`: Models with different speeds
- `training.csv`, `testing.csv`: Sample flow-based data
- `requirements.txt`: Dependencies

---

## 🚀 How to Run

```bash
# 1. Create virtual env (recommended)
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit app
streamlit run app.py
