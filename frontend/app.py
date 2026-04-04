import streamlit as st
import requests
from datetime import datetime
import json


st.set_page_config(
    page_title="NayaySahayak - Virtual Senior Prosecutor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 5px;
        color: #155724;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 5px;
        color: #721c24;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        padding: 1rem;
        border-radius: 5px;
        color: #856404;
    }
    </style>
""", unsafe_allow_html=True)


API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")
API_ENDPOINT = f"{API_BASE_URL}/api/v1/analyze-case-rag"

def main():
    
    st.markdown('<div class="main-header">⚖️ NyayShayak</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Virtual Senior Prosecutor - AI-Powered Legal Case Analysis</div>', unsafe_allow_html=True)
    
    
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        **NyayShayak** is an AI-powered legal assistant that helps prosecutors analyze case documents.
        
        **Features:**
        - 📄 OCR for Hindi documents
        - 🔄 Automatic translation
        - 🤖 AI-powered case analysis
        - 📊 Offense identification
        - 🔍 Evidence gap detection
        """)
        
        st.header("⚙️ Settings")
        api_url = st.text_input("Backend API URL", value=API_BASE_URL)
        
        st.header("📊 System Status")
        if st.button("Check Server Status"):
            try:
                response = requests.get(f"{api_url}/", timeout=5)
                if response.status_code == 200:
                    st.success("✅ Backend server is running")
                else:
                    st.error(f"❌ Server returned status code: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Cannot connect to backend: {str(e)}")
    
    
    st.header("📁 Upload Case Document")
    
    
    uploaded_file = st.file_uploader(
        "Upload a PDF or Image file (Hindi/English)",
        type=["pdf", "jpg", "jpeg", "png"],
        help="Supported formats: PDF, JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("File Type", uploaded_file.type)
        with col3:
            file_size = uploaded_file.size / 1024  # Convert to KB
            st.metric("File Size", f"{file_size:.2f} KB")
        
       
        if uploaded_file.type.startswith("image"):
            with st.expander("🖼️ Image Preview"):
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
       
        if st.button("🔍 Analyze Case", type="primary", use_container_width=True):
            analyze_case(uploaded_file, api_url)

def analyze_case(uploaded_file, api_url):
    """Send file to backend API and display results"""
    
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        
        status_text.text("📤 Uploading file to server...")
        progress_bar.progress(10)
        
        uploaded_file.seek(0)
        
        
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        
        
        status_text.text("🔄 Processing document (OCR + Translation + Analysis)...")
        progress_bar.progress(30)
        
        
        response = requests.post(
            f"{api_url}/api/v1/analyze-case-rag",
            files=files,
            timeout=18000  
        )
        
        progress_bar.progress(90)
        
        
        if response.status_code == 200:
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
           
            result = response.json()
            
            
            display_results(result)
            
        else:
           
            progress_bar.empty()
            status_text.empty()
            
            error_detail = "Unknown error"
            try:
                error_data = response.json()
                error_detail = error_data.get("detail", str(error_data))
            except:
                error_detail = response.text
            
            st.markdown(f'<div class="error-box">❌ <strong>Error:</strong> {error_detail}</div>', unsafe_allow_html=True)
    
    except requests.exceptions.Timeout:
        progress_bar.empty()
        status_text.empty()
        st.markdown('''<div class="error-box">
            ⏱️ <strong>Timeout Error:</strong> The analysis is taking longer than expected (>25 minutes).<br>
            <strong>What's happening:</strong> The backend is still processing your request, but the frontend stopped waiting.<br>
            <strong>Suggestions:</strong>
            <ul>
                <li>Try with a smaller/simpler document first</li>
                <li>The backend may still complete - check backend logs</li>
                <li>Consider implementing an async job queue for very large files</li>
            </ul>
        </div>''', unsafe_allow_html=True)
    
    except requests.exceptions.ConnectionError:
        progress_bar.empty()
        status_text.empty()
        st.markdown('<div class="error-box">🔌 <strong>Connection Error:</strong> Cannot connect to the backend server. Make sure it\'s running.</div>', unsafe_allow_html=True)
    
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.markdown(f'<div class="error-box">❌ <strong>Unexpected Error:</strong> {str(e)}</div>', unsafe_allow_html=True)

def display_results(result):
    
    
    st.success("✅ Case analysis completed successfully!")
    
    # Case ID
    st.markdown("---")
    st.subheader("📋 Case Information")
    st.code(f"Case ID: {result.get('case_id', 'N/A')}", language=None)
    
    # Summary
    st.markdown("---")
    st.subheader("📝 Case Summary")
    st.markdown(f'<div class="result-box">{result.get("summary", "No summary available")}</div>', unsafe_allow_html=True)
    
    # Offenses
    st.markdown("---")
    st.subheader("⚠️ Identified Offenses")
    offenses = result.get("offenses", [])
    if offenses:
        for i, offense in enumerate(offenses, 1):
            st.markdown(f"**{i}.** {offense}")
    else:
        st.info("No offenses identified in this case.")
    
    # Missing Evidence
    st.markdown("---")
    st.subheader("🔍 Missing Evidence / Gaps")
    missing_evidence = result.get("missing_evidence", [])
    if missing_evidence:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        for i, evidence in enumerate(missing_evidence, 1):
            st.markdown(f"**{i}.** {evidence}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.success("✅ No critical evidence gaps identified.")
    
    # Recommendation
    st.markdown("---")
    st.subheader("💡 Recommendation")
    st.markdown(f'<div class="result-box"><strong>{result.get("recommendation", "No recommendation available")}</strong></div>', unsafe_allow_html=True)
    
    # Export options
    st.markdown("---")
    st.subheader("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        
        json_str = json.dumps(result, indent=2)
        st.download_button(
            label="📥 Download as JSON",
            data=json_str,
            file_name=f"case_{result.get('case_id', 'unknown')}.json",
            mime="application/json"
        )
    
    with col2:
       
        text_report = generate_text_report(result)
        st.download_button(
            label="📥 Download as Text",
            data=text_report,
            file_name=f"case_{result.get('case_id', 'unknown')}.txt",
            mime="text/plain"
        )

def generate_text_report(result):
    """Generate a text report from the analysis results"""
    report = f"""
NyayShayak - Case Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'=' * 60}

CASE ID: {result.get('case_id', 'N/A')}

SUMMARY
{'-' * 60}
{result.get('summary', 'No summary available')}

IDENTIFIED OFFENSES
{'-' * 60}
"""
    offenses = result.get('offenses', [])
    if offenses:
        for i, offense in enumerate(offenses, 1):
            report += f"{i}. {offense}\n"
    else:
        report += "No offenses identified.\n"
    
    report += f"""
MISSING EVIDENCE / GAPS
{'-' * 60}
"""
    missing_evidence = result.get('missing_evidence', [])
    if missing_evidence:
        for i, evidence in enumerate(missing_evidence, 1):
            report += f"{i}. {evidence}\n"
    else:
        report += "No critical evidence gaps identified.\n"
    
    report += f"""
RECOMMENDATION
{'-' * 60}
{result.get('recommendation', 'No recommendation available')}

{'=' * 60}
End of Report
"""
    return report

if __name__ == "__main__":
    main()
