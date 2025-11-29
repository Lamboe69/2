#!/usr/bin/env python3
"""
Streamlit Web App for Graph-Reasoned USL System
Optimized for Render deployment
"""

import streamlit as st
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main system
from complete_usl_system import GraphReasonedUSLApp

def main():
    """Main entry point for Streamlit app"""
    
    # Configure Streamlit for production
    st.set_page_config(
        page_title="USL Translation System",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize and run the app
    app = GraphReasonedUSLApp()
    app.run()

if __name__ == "__main__":
    main()