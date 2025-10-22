# UTC Factbook: State of Residence (Single Term)

A Streamlit app that renders a UTC-branded Plotly choropleth by **state of residence** for a selected academic term. Built for portfolio + lightweight internal use.

![Screenshot](images/screenshot.png)

## Features
- Single-term map (e.g., Spring 2025) with UTC Gold→Blue colors
- Auto-contrast labels (white on dark, black on light)
- Sidebar controls (label size, min count, exclude AK/HI/PR)
- Top-N states table
- Export **HTML** and **PNG**
- Handles `TermYear` as `202520` **or** `"Spring 2025"`

## Install & Run (local)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
