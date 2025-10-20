# D:\new project live imple\trustworthy\run_streamlit.py
import sys
import os

# Path to your app file
APP = r"D:\new project live imple\trustworthy\app_streamlit.py"

# Change directory to the folder of your project
os.chdir(os.path.dirname(D:\new project live imple\trustworthy))

# Import streamlit launcher
try:
    from streamlit.web import cli as stcli  # Streamlit >= 1.12
except Exception:
    import streamlit.cli as stcli           # older fallback

# Run the Streamlit app
sys.argv = ["streamlit", "run", APP, "--server.headless=false"]
sys.exit(stcli.main())

