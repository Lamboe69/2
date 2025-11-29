#!/usr/bin/env python3
"""
USL Clinical Screening System - Professional Healthcare Interface
Complete web application for Ugandan Sign Language infectious disease screening

Features:
- Professional healthcare UI design
- Real-time video processing with all 3 models
- Clinical workflow management
- FHIR integration and export
- Advanced analytics and reporting
- Optimized for Render deployment
"""

import streamlit as st
import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import base64
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from usl_inference import USLInferencePipeline
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Model inference not available: {e}")
    MODELS_AVAILABLE = False
    USLInferencePipeline = None

# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="USL Clinical Screening - Uganda",
    page_icon="🇺🇬",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/your-org/usl-clinical-screening',
        'Report a bug': 'https://github.com/your-org/usl-clinical-screening/issues',
        'About': '''
        ## USL Clinical Screening System
        **Built for Ugandan Healthcare** 🇺🇬
        Real-time sign language processing for infectious disease screening.
        '''
    }
)

# Custom CSS for professional healthcare styling
st.markdown("""
<style>
    /* Professional healthcare color scheme */
    :root {
        --primary-color: #2E8B57;
        --secondary-color: #FF6B35;
        --accent-color: #1E3A8A;
        --background-color: #F8FAFC;
        --card-background: #FFFFFF;
        --text-primary: #1E293B;
        --text-secondary: #64748B;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --danger-color: #EF4444;
    }

    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(46, 139, 87, 0.2);
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .main-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }

    /* Card styling */
    .metric-card {
        background: var(--card-background);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid var(--primary-color);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        margin-bottom: 1rem;
    }

    .status-card {
        background: var(--card-background);
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid var(--success-color);
        margin: 0.5rem 0;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(46, 139, 87, 0.3);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 139, 87, 0.4);
    }

    .danger-button>button {
        background: linear-gradient(135deg, var(--danger-color), #DC2626);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: var(--background-color);
        padding: 1rem;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--text-secondary);
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: var(--primary-color);
        color: white;
        box-shadow: 0 4px 15px rgba(46, 139, 87, 0.3);
    }

    /* Sidebar styling */
    .sidebar-header {
        background: linear-gradient(135deg, var(--accent-color), var(--primary-color));
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
    }

    /* Progress indicators */
    .progress-bar {
        width: 100%;
        height: 8px;
        background: #E5E7EB;
        border-radius: 4px;
        overflow: hidden;
        margin: 1rem 0;
    }

    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--success-color), var(--primary-color));
        transition: width 0.3s ease;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    .status-success {
        background: rgba(16, 185, 129, 0.1);
        color: var(--success-color);
        border: 1px solid var(--success-color);
    }

    .status-warning {
        background: rgba(245, 158, 11, 0.1);
        color: var(--warning-color);
        border: 1px solid var(--warning-color);
    }

    .status-danger {
        background: rgba(239, 68, 68, 0.1);
        color: var(--danger-color);
        border: 1px solid var(--danger-color);
    }

    /* Animation for loading */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .loading-pulse {
        animation: pulse 2s infinite;
    }

    /* Footer styling */
    .footer {
        background: var(--text-primary);
        color: white;
        padding: 2rem;
        text-align: center;
        margin-top: 3rem;
        border-radius: 15px 15px 0 0;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN HEADER
# ============================================================================

st.markdown("""
<div class="main-header">
    <h1 class="main-title">🇺🇬 USL Clinical Screening System</h1>
    <p class="main-subtitle">Real-time Ugandan Sign Language Processing for Infectious Disease Screening</p>
    <p style="font-size: 1rem; margin-top: 1rem;">
        🏥 Built for Ugandan Healthcare • 🤟 Powered by Sign Language • 📊 WHO Guidelines Compliant
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None

if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

if 'current_patient' not in st.session_state:
    st.session_state.current_patient = {
        'id': 'PATIENT-001',
        'completed_slots': [],
        'danger_signs': []
    }

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_models():
    """Load trained USL models"""
    try:
        pipeline = USLInferencePipeline(
            sign_model_path='./usl_models/sign_recognition_model.pth',
            screening_model_path='./usl_models/usl_screening_model.pth',
            sign_vocab_path='./usl_models/sign_vocabulary.json',
            device='cpu'  # Use CPU for better compatibility in web apps
        )
        return pipeline
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.error("Please ensure model files are in the ./usl_models/ directory")
        import traceback
        st.error(traceback.format_exc())
        return None

# Load models
if st.session_state.pipeline is None:
    with st.spinner("🔄 Loading USL models..."):
        st.session_state.pipeline = load_models()

pipeline = st.session_state.pipeline

if pipeline is None:
    st.error("❌ Failed to load models. Please check the model files.")
    st.stop()

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

with st.sidebar:
    st.header("👤 Patient Information")

    patient_id = st.text_input("Patient ID", st.session_state.current_patient['id'])

    col1, col2 = st.columns(2)
    with col1:
        patient_age = st.number_input("Age", min_value=0, max_value=120, value=25)
    with col2:
        patient_gender = st.selectbox("Gender", ["Female", "Male", "Other"])

    patient_location = st.text_input("Location", "Kampala")

    # Update patient info
    if patient_id != st.session_state.current_patient['id']:
        st.session_state.current_patient['id'] = patient_id
        st.session_state.current_patient['completed_slots'] = []
        st.session_state.current_patient['danger_signs'] = []

    st.header("📊 System Status")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sign Vocabulary", len(pipeline.sign_vocab))
    with col2:
        st.metric("Screening Slots", len(pipeline.screening_slots))

    st.metric("Completed Analyses", len(st.session_state.analysis_history))

    st.header("⚠️ Danger Signs Monitor")

    if st.session_state.current_patient['danger_signs']:
        st.error("🚨 DANGER SIGNS DETECTED!")
        for sign in st.session_state.current_patient['danger_signs']:
            st.write(f"• {sign}")
    else:
        st.success("✅ No danger signs detected")

# ============================================================================
# MAIN CONTENT
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["🎥 Video Analysis", "📋 Clinical Workflow", "📊 Analytics", "📄 FHIR Export"])

# ============================================================================
# TAB 1: VIDEO ANALYSIS
# ============================================================================

with tab1:
    st.header("Real-time Video Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Upload Patient Video")
        uploaded_file = st.file_uploader(
            "Choose a video file (MP4, AVI, MOV)",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video of the patient signing about their symptoms"
        )

        if uploaded_file is not None:
            # Save uploaded file temporarily
            video_path = f"./temp_{uploaded_file.name}"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"✅ Video uploaded: {uploaded_file.name}")

            # Analysis button
            if st.button("🔍 Analyze Video", key="analyze_video", type="primary"):
                with st.spinner("🎥 Processing video... This may take a moment."):
                    try:
                        # Process video
                        result = pipeline.process_video(video_path)

                        # Store result
                        st.session_state.last_result = result
                        st.session_state.analysis_history.append(result)

                        # Update patient screening progress
                        screening_slot = result['screening']['screening_slot']
                        if screening_slot not in st.session_state.current_patient['completed_slots']:
                            st.session_state.current_patient['completed_slots'].append(screening_slot)

                        # Check for danger signs
                        danger_assessment = pipeline.detect_danger_signs(result)
                        if danger_assessment['danger_detected']:
                            st.session_state.current_patient['danger_signs'].extend(danger_assessment['danger_signs'])

                        st.success("✅ Analysis complete!")

                    except Exception as e:
                        st.error(f"❌ Analysis failed: {e}")
                        import traceback
                        st.error(traceback.format_exc())
                    finally:
                        # Clean up
                        if os.path.exists(video_path):
                            os.remove(video_path)

    with col2:
        st.subheader("Analysis Status")
        if 'last_result' in st.session_state:
            result = st.session_state.last_result
            st.metric("Frames Processed", result['pose_frames'])
            st.metric("Signs Detected", result['signs']['num_signs'])

            confidence = result['screening']['confidence']
            if confidence > 0.8:
                st.success(f"High Confidence: {confidence:.1%}")
            elif confidence > 0.6:
                st.warning(f"Medium Confidence: {confidence:.1%}")
            else:
                st.error(f"Low Confidence: {confidence:.1%}")

    # Display results
    if 'last_result' in st.session_state:
        result = st.session_state.last_result

        st.divider()
        st.subheader("📊 Analysis Results")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🤟 Recognized Signs")
            signs_df = pd.DataFrame({
                'Sign': result['signs']['sign_names'][:10],  # Show first 10
                'Confidence': [f"{result['signs']['confidence']:.1%}"] * len(result['signs']['sign_names'][:10])
            })
            st.dataframe(signs_df, use_container_width=True)

            if len(result['signs']['sign_names']) > 10:
                st.info(f"And {len(result['signs']['sign_names']) - 10} more signs...")

        with col2:
            st.subheader("🏥 Screening Classification")
            screening = result['screening']

            # Main result
            st.metric(
                f"Question: {screening['screening_slot'].replace('_', ' ').title()}",
                f"Answer: {screening['response'].upper()}",
                f"{screening['confidence']:.1%}"
            )

            # Confidence visualization
            fig = go.Figure(data=[
                go.Bar(
                    x=['Confidence'],
                    y=[screening['confidence']],
                    marker_color='lightblue',
                    text=[f"{screening['confidence']:.1%}"],
                    textposition='auto'
                )
            ])
            fig.update_layout(
                title="Prediction Confidence",
                yaxis_range=[0, 1],
                height=200,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # Danger signs alert
        danger_assessment = pipeline.detect_danger_signs(result)
        if danger_assessment['danger_detected']:
            st.error("🚨 CRITICAL: Danger Signs Detected!")
            st.write("**Immediate medical attention required!**")
            for sign in danger_assessment['danger_signs']:
                st.write(f"• {sign.replace('_', ' ').title()}")

# ============================================================================
# TAB 2: CLINICAL WORKFLOW
# ============================================================================

with tab2:
    st.header("Clinical Screening Workflow")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Screening Progress")

        # Display completed slots
        completed = st.session_state.current_patient['completed_slots']
        all_slots = pipeline.screening_slots

        progress_data = []
        for slot in all_slots:
            status = "✅ Completed" if slot in completed else "⏳ Pending"
            progress_data.append({
                'Screening Question': slot.replace('_', ' ').title(),
                'Status': status,
                'Priority': 'High' if slot in ['fever', 'danger_signs'] else 'Normal'
            })

        progress_df = pd.DataFrame(progress_data)
        st.dataframe(progress_df, use_container_width=True, hide_index=True)

        # Next recommended question
        next_slot = pipeline.implement_skip_logic(completed)
        if next_slot:
            st.info(f"📋 **Next Recommended Question:** {next_slot.replace('_', ' ').title()}")
        else:
            st.success("🎉 **Screening Complete!** All questions answered.")

    with col2:
        st.subheader("Patient Summary")

        st.metric("Patient ID", patient_id)
        st.metric("Age", patient_age)
        st.metric("Gender", patient_gender)
        st.metric("Location", patient_location)
        st.metric("Slots Completed", len(completed))

        # Risk assessment
        risk_level = "High" if st.session_state.current_patient['danger_signs'] else "Low"
        risk_color = "🔴" if risk_level == "High" else "🟢"

        st.metric("Risk Level", f"{risk_color} {risk_level}")

        if st.button("🔄 Reset Patient", key="reset_patient"):
            st.session_state.current_patient = {
                'id': f'PATIENT-{int(time.time())}',
                'completed_slots': [],
                'danger_signs': []
            }
            st.rerun()

# ============================================================================
# TAB 3: ANALYTICS
# ============================================================================

with tab3:
    st.header("System Analytics & Insights")

    if st.session_state.analysis_history:
        col1, col2, col3, col4 = st.columns(4)

        history = st.session_state.analysis_history

        with col1:
            avg_confidence = np.mean([r['screening']['confidence'] for r in history])
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")

        with col2:
            total_signs = sum([r['signs']['num_signs'] for r in history])
            st.metric("Total Signs", total_signs)

        with col3:
            unique_slots = len(set([r['screening']['screening_slot'] for r in history]))
            st.metric("Unique Questions", unique_slots)

        with col4:
            danger_count = sum([1 for r in history if pipeline.detect_danger_signs(r)['danger_detected']])
            st.metric("Danger Detections", danger_count)

        st.divider()

        # Screening distribution
        st.subheader("Screening Questions Distribution")

        slot_counts = {}
        for result in history:
            slot = result['screening']['screening_slot']
            slot_counts[slot] = slot_counts.get(slot, 0) + 1

        if slot_counts:
            fig = px.bar(
                x=list(slot_counts.keys()),
                y=list(slot_counts.values()),
                labels={'x': 'Screening Question', 'y': 'Count'},
                title="Screening Questions Asked"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

        # Response distribution
        st.subheader("Patient Responses")

        response_counts = {'yes': 0, 'no': 0, 'unknown': 0}
        for result in history:
            response = result['screening']['response']
            if response in response_counts:
                response_counts[response] += 1

        fig = px.pie(
            values=list(response_counts.values()),
            names=list(response_counts.keys()),
            title="Response Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Recent analyses table
        st.subheader("Recent Analyses")

        recent_data = []
        for i, result in enumerate(history[-10:]):  # Last 10
            recent_data.append({
                'Analysis #': len(history) - 9 + i,
                'Signs Detected': result['signs']['num_signs'],
                'Question': result['screening']['screening_slot'].replace('_', ' ').title(),
                'Response': result['screening']['response'].upper(),
                'Confidence': f"{result['screening']['confidence']:.1%}",
                'Timestamp': result['timestamp'][:19]
            })

        if recent_data:
            recent_df = pd.DataFrame(recent_data)
            st.dataframe(recent_df, use_container_width=True, hide_index=True)

    else:
        st.info("📊 No analyses yet. Upload a video to get started!")

# ============================================================================
# TAB 4: FHIR EXPORT
# ============================================================================

with tab4:
    st.header("Clinical Data Export (FHIR)")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Patient Information")

        col_a, col_b = st.columns(2)
        with col_a:
            export_patient_id = st.text_input("Patient ID", patient_id, key="export_patient_id")
            export_age = st.number_input("Age", min_value=0, max_value=120, value=patient_age, key="export_age")
        with col_b:
            export_gender = st.selectbox("Gender", ["Female", "Male", "Other"], index=["Female", "Male", "Other"].index(patient_gender), key="export_gender")
            export_location = st.text_input("Location", patient_location, key="export_location")

    with col2:
        st.subheader("Export Summary")
        if st.session_state.analysis_history:
            st.metric("Available Results", len(st.session_state.analysis_history))
            st.metric("Patient ID", export_patient_id)

            # Calculate overall risk
            danger_count = sum([1 for r in st.session_state.analysis_history
                              if pipeline.detect_danger_signs(r)['danger_detected']])
            risk_level = "High" if danger_count > 0 else "Low"
            st.metric("Risk Assessment", risk_level)
        else:
            st.info("No results to export yet")

    st.divider()

    # Generate FHIR bundle
    if st.button("📄 Generate FHIR Bundle", key="generate_fhir", type="primary"):
        if st.session_state.analysis_history:
            # Use the most recent result for primary observation
            latest_result = st.session_state.analysis_history[-1]

            # Create comprehensive FHIR bundle
            fhir_bundle = {
                "resourceType": "Bundle",
                "type": "collection",
                "timestamp": datetime.now().isoformat(),
                "entry": []
            }

            # Add patient resource
            patient_resource = {
                "resource": {
                    "resourceType": "Patient",
                    "id": export_patient_id,
                    "identifier": [{"value": export_patient_id}],
                    "gender": export_gender.lower(),
                    "birthDate": f"{datetime.now().year - export_age}-01-01",  # Approximate
                    "address": [{"text": export_location}]
                }
            }
            fhir_bundle["entry"].append({"resource": patient_resource["resource"]})

            # Add observations for each analysis
            for i, result in enumerate(st.session_state.analysis_history):
                observation = {
                    "resource": {
                        "resourceType": "Observation",
                        "id": f"obs-{i+1}",
                        "status": "final",
                        "code": {
                            "coding": [{
                                "system": "http://medisign.ug/screening",
                                "code": result['screening']['screening_slot'],
                                "display": result['screening']['screening_slot'].replace('_', ' ').title()
                            }]
                        },
                        "valueString": result['screening']['response'],
                        "extension": [{
                            "url": "http://medisign.ug/confidence",
                            "valueDecimal": result['screening']['confidence']
                        }, {
                            "url": "http://medisign.ug/signs",
                            "valueString": ", ".join(result['signs']['sign_names'])
                        }],
                        "subject": {
                            "reference": f"Patient/{export_patient_id}"
                        },
                        "effectiveDateTime": result['timestamp']
                    }
                }
                fhir_bundle["entry"].append({"resource": observation["resource"]})

            # Display FHIR bundle
            st.subheader("📋 Generated FHIR Bundle")
            st.json(fhir_bundle)

            # Download button
            bundle_json = json.dumps(fhir_bundle, indent=2)
            st.download_button(
                label="💾 Download FHIR Bundle",
                data=bundle_json,
                file_name=f"usl_screening_{export_patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        else:
            st.warning("No analysis results available. Please analyze a video first.")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**System**: ✅ USL Clinical Screening")
with col2:
    st.markdown("**Models**: ✅ Loaded & Ready")
with col3:
    st.markdown("**Training**: ✅ Real Dataset Models")
with col4:
    st.markdown(f"**Last Updated**: {datetime.now().strftime('%H:%M:%S')}")

# ============================================================================
# DEBUG INFORMATION (Hidden)
# ============================================================================

if st.checkbox("🔧 Show Debug Info", key="debug_checkbox"):
    st.subheader("Debug Information")

    st.write("**Session State:**")
    st.json({
        'pipeline_loaded': st.session_state.pipeline is not None,
        'history_length': len(st.session_state.analysis_history),
        'current_patient': st.session_state.current_patient
    })

    if 'last_result' in st.session_state:
        st.write("**Last Result:**")
        st.json(st.session_state.last_result)

    st.write("**Model Info:**")
    if pipeline:
        st.json({
            'sign_vocab_size': len(pipeline.sign_vocab),
            'screening_slots': len(pipeline.screening_slots),
            'device': str(pipeline.device)
        })
