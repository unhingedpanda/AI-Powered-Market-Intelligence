"""
AI Market Intelligence System - Clean Deliverable-Focused Interface
Main Streamlit application with clear deliverable mapping.
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path
import sys
import re
from typing import Optional

# Add src to path for imports
sys.path.append('src')

from analysis.insights import GeminiInsightsGenerator
from data_pipeline.loader import load_google_play_data, load_d2c_dataset
from data_pipeline.cleaner import clean_google_play_data, clean_d2c_data, create_unified_schema, create_sample_app_store_data
from api_integration.app_store import AppStoreClient

# Configure page
st.set_page_config(
    page_title="AI Market Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean CSS - theme compatible
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        color: white;
        border-radius: 0.75rem;
        margin-bottom: 1.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }
    
    .deliverable-card {
        background: var(--background-color);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }

    .neutral-card {
        background: rgba(255,255,255,0.95);
        border: 1px solid rgba(226,232,240,0.9);
        border-radius: 0.75rem;
        padding: 1rem;
        color: #1f2937;
    }

    .neutral-card .neutral-subtext,
    .neutral-card p,
    .neutral-card li,
    .neutral-card span {
        color: #475569;
    }

    body[data-theme="dark"] .neutral-card {
        background: rgba(30,41,59,0.7);
        border-color: rgba(148,163,184,0.45);
        color: #e2e8f0;
    }

    body[data-theme="dark"] .neutral-card .neutral-subtext,
    body[data-theme="dark"] .neutral-card p,
    body[data-theme="dark"] .neutral-card li,
    body[data-theme="dark"] .neutral-card span {
        color: #cbd5f5;
    }
</style>
""", unsafe_allow_html=True)


def load_data():
    """Load and cache all data."""
    @st.cache_data
    def load_processed_data():
        try:
            # Load cleaned Google Play data
            gp_data = pd.read_csv("data/processed/googleplaystore_cleaned.csv")
            
            # Load unified data
            try:
                unified_data = pd.read_csv("data/processed/unified_data.csv")
            except FileNotFoundError:
                # Create unified data if it doesn't exist
                from data_pipeline.cleaner import create_unified_schema
                unified_data = create_unified_schema(gp_data)
                unified_data.to_csv("data/processed/unified_data.csv", index=False)

            # Load D2C data
            d2c_data = pd.read_csv("data/processed/d2c_cleaned.csv")

            return gp_data, unified_data, d2c_data

        except FileNotFoundError as e:
            st.error(f"❌ Data file not found: {e}")
            return None, None, None
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            return None, None, None

    return load_processed_data()


def load_insights():
    """Load AI-generated insights."""
    @st.cache_data
    def load_insights_data():
        insights = {}
        insights_dir = Path("data/processed")
        insight_files = list(insights_dir.glob("*insights*.json"))

        for file_path in insight_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Extract insights from metadata structure
                if 'insights' in data:
                    insights[file_path.stem] = data['insights']
                else:
                    insights[file_path.stem] = data

            except Exception as e:
                st.warning(f"Could not load {file_path.name}: {e}")

        return insights

    return load_insights_data()


def fetch_reviews_for_app(
    client: AppStoreClient,
    app_id: str,
    country: str,
    sort: str,
    lang: str,
    target_count: int,
    status_callback=None,
    start_page: int = 1,
    max_pages: int = 10
) -> tuple[list[dict], int, int, int]:
    """Fetch paginated reviews until target_count reached.

    Returns (reviews, next_page, pages_fetched, page_size_estimate).
    """
    collected: list[dict] = []
    page = start_page
    pages_fetched = 0
    page_size_estimate = 0

    while len(collected) < target_count and pages_fetched < max_pages:
        if status_callback:
            status_callback(page)

        page_reviews = client.get_reviews(
            app_id,
            country=country,
            page=page,
            sort=sort,
            lang=lang
        )

        if not page_reviews:
            break

        collected.extend(page_reviews)
        pages_fetched += 1

        if page_size_estimate == 0:
            page_size_estimate = len(page_reviews)

        # If the API returns fewer results than a full page, we assume it's the end
        if len(page_reviews) < max(page_size_estimate, 1):
            page += 1
            break

        page += 1

    return collected[:target_count], page, pages_fetched, page_size_estimate


def download_data_files(unified_data: pd.DataFrame, d2c_data: pd.DataFrame):
    """Render download buttons for primary data files."""
    if unified_data is not None:
        csv_data = unified_data.to_csv(index=False)
        st.download_button(
            label="📄 Unified Dataset (CSV)",
            data=csv_data,
            file_name="deliverable_1_unified_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Unified dataset not available. Re-run the data pipeline.")

    if d2c_data is not None:
        d2c_csv = d2c_data.to_csv(index=False)
        st.download_button(
            label="🎯 D2C Dataset (CSV)",
            data=d2c_csv,
            file_name="d2c_campaigns_with_metrics.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("D2C dataset not available.")


def download_insights_files(insights: dict):
    """Render download buttons for insight JSON files."""
    if not insights:
        st.info("No insights available. Generate new insights via the Live AI Generator.")
        return

    for insight_name, insight_data in insights.items():
        insight_json = json.dumps(insight_data, indent=2, ensure_ascii=False)
        st.download_button(
            label=f"📄 {insight_name}",
            data=insight_json,
            file_name=f"deliverable_2_{insight_name}.json",
            mime="application/json",
            key=f"download_{insight_name}",
            use_container_width=True
        )


def download_report_and_docs():
    """Render download buttons for reports and documentation."""
    report_found = False

    html_report_path = Path("reports/executive_report.html")
    if html_report_path.exists():
        with html_report_path.open('r', encoding='utf-8') as f:
            report_content = f.read()
        st.download_button(
            label="📄 Executive Report (HTML)",
            data=report_content,
            file_name="deliverable_3_executive_report.html",
            mime="text/html",
            use_container_width=True
        )
        report_found = True

    markdown_reports = sorted(Path("reports").glob("executive_summary_*.md")) if Path("reports").exists() else []
    if markdown_reports:
        latest_md = markdown_reports[-1]
        with latest_md.open('r', encoding='utf-8') as f:
            report_md_content = f.read()
        st.download_button(
            label="📄 Executive Summary (Markdown)",
            data=report_md_content,
            file_name=latest_md.name,
            mime="text/markdown",
            use_container_width=True
        )
        report_found = True

    if not report_found:
        st.info("Generate an executive report from the 📈 page before downloading.")

    readme_path = Path("README.md")
    if readme_path.exists():
        with readme_path.open('r', encoding='utf-8') as f:
            readme_content = f.read()
        st.download_button(
            label="📄 README.md",
            data=readme_content,
            file_name="README.md",
            mime="text/markdown",
            use_container_width=True
        )

def _display_insight_details(insight_data, insight_name):
    """Display detailed insight information with improved formatting."""
    
    # Handle nested data structure
    if isinstance(insight_data, dict) and 'insights' in insight_data:
        insight_data = insight_data['insights']
    
    # Basic metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'total_reviews_analyzed' in insight_data:
            st.metric("Reviews Analyzed", insight_data['total_reviews_analyzed'])
        elif 'total_campaigns' in insight_data:
            st.metric("Campaigns Analyzed", insight_data['total_campaigns'])
    
    with col2:
        if 'average_rating' in insight_data:
            rating = insight_data.get('average_rating', 0)
            st.metric("Average Rating", f"{rating:.2f}/5")
    
    with col3:
        if 'category' in insight_data:
            st.metric("Category", insight_data['category'])
        elif 'seo_category' in insight_data:
            st.metric("SEO Category", insight_data['seo_category'])

    # Sentiment summary
    if 'sentiment_summary' in insight_data:
        st.subheader("💭 Sentiment Analysis")
        st.write(insight_data['sentiment_summary'])

    # Features with better formatting
    features = insight_data.get('requested_features', [])
    if features:
        st.subheader("🏆 Top Requested Features")
        for i, feature in enumerate(features[:3], 1):
            with st.expander(f"{i}. {feature.get('feature', 'N/A')}", expanded=False):
                st.write(f"**Confidence:** {feature.get('confidence_score', 0):.1%}")
                if 'evidence' in feature:
                    st.write(f"**Evidence:** {feature['evidence']}")
                if 'recommendation' in feature:
                    st.write(f"**Recommendation:** {feature['recommendation']}")

    # Complaints with better formatting
    complaints = insight_data.get('common_complaints', [])
    if complaints:
        st.subheader("⚠️ Common Issues")
        for i, complaint in enumerate(complaints[:3], 1):
            complaint_text = complaint.get('complaint', complaint.get('lack_of_value', complaint.get('performance_issues', 'N/A')))
            with st.expander(f"{i}. {complaint_text}", expanded=False):
                st.write(f"**Confidence:** {complaint.get('confidence_score', 0):.1%}")
                if 'evidence' in complaint:
                    st.write(f"**Evidence:** {complaint['evidence']}")
                if 'recommendation' in complaint:
                    st.write(f"**Recommendation:** {complaint['recommendation']}")
    
    # D2C specific content
    if 'top_performers' in insight_data:
        st.subheader("🏆 Top Performing Campaigns")
        for performer in insight_data['top_performers'][:3]:
            with st.expander(f"{performer.get('campaign_name', 'N/A')} - ROAS: {performer.get('roas', 0):.2f}x"):
                st.write(f"**Conversion Rate:** {performer.get('conversion_rate', 0):.2f}%")
                st.write(f"**Success Factors:** {performer.get('success_factors', 'N/A')}")
    
    if 'creative_content' in insight_data:
        creative = insight_data['creative_content']
        st.subheader("🎨 Generated Creative Content")
        
        if 'ad_headlines' in creative and creative['ad_headlines']:
            st.write("**Ad Headlines:**")
            for i, headline in enumerate(creative['ad_headlines'], 1):
                st.write(f"{i}. {headline}")
        
        if 'meta_description' in creative and creative['meta_description']:
            st.write(f"**Meta Description:** {creative['meta_description']}")
    
    # Download button for this specific insight
    insight_json = json.dumps(insight_data, indent=2, ensure_ascii=False)
    st.download_button(
        label="📄 Download This Insight",
        data=insight_json,
        file_name=f"{insight_name}_insight.json",
        mime="application/json",
        help="Download this specific insight as JSON"
    )


def main():
    """Main application with clean deliverable-focused design."""

    # Clean header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Market Intelligence System</h1>
        <p>Applied AI Engineer Assignment - Deliverable Demonstration</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner("🚀 Loading system..."):
        gp_data, unified_data, d2c_data = load_data()
        insights = load_insights()

    if gp_data is None:
        st.error("❌ System unavailable. Please check data files.")
        return

    # Clean sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1rem; background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
                    border-radius: 1rem; color: white; margin-bottom: 1rem;">
            <h3 style="margin: 0; text-align: center;">📋 Deliverables</h3>
        </div>
        """, unsafe_allow_html=True)

        # Clean navigation focused on deliverables
        nav_items = [
            ("🏠 Overview", "Project introduction and assignment overview"),
            ("📊 Data Pipeline Results", "Deliverable #1: Clean combined datasets"),
            ("📈 Executive Reports", "Deliverable #3: Automated analysis reports"),
            ("🤖 Live AI Generator", "Deliverable #4: CLI/Streamlit interface to query insights"),
            ("📦 Deliverables Summary", "All submission files and downloads")
        ]

        current_page = st.session_state.get('page', '🏠 Overview')
        
        for page_name, description in nav_items:
            button_type = "primary" if page_name == current_page else "secondary"
            
            if st.button(
                page_name,
                help=description,
                use_container_width=True,
                type=button_type
            ):
                st.session_state.page = page_name
                st.rerun()

        st.markdown("---")
        st.caption("💡 Navigate through all 5 deliverables")

    # Quick access buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 View Data Pipeline", help="Deliverable #1: Clean datasets", use_container_width=True):
            st.session_state.page = "📊 Data Pipeline Results"
    with col2:
        if st.button("📈 View Reports", help="Deliverable #3: Executive reports", use_container_width=True):
            st.session_state.page = "📈 Executive Reports"
    with col3:
        if st.button("🤖 Query Insights", type="primary", help="Deliverable #4: Live interface", use_container_width=True):
            st.session_state.page = "🤖 Live AI Generator"

    # Clean page routing
    page = st.session_state.get('page', '🏠 Overview')

    if page == "🏠 Overview":
        show_overview()
    elif page == "📊 Data Pipeline Results":
        show_data_pipeline_results(gp_data, unified_data, d2c_data)
    elif page == "📈 Executive Reports":
        show_executive_reports()
    elif page == "🤖 Live AI Generator":
        show_live_ai_generator()
    elif page == "📦 Deliverables Summary":
        show_deliverables_summary(gp_data, unified_data, d2c_data, insights)


def show_overview():
    """Show clean project overview."""
    
    # Assignment Overview
    st.markdown("### 🎯 Applied AI Engineer Assignment")
    st.markdown("""
    Build an AI-powered market intelligence system that ingests multiple data sources, generates LLM insights, and produces deliverable-ready outputs.
    """)
    
    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem;">
        <div class="neutral-card">
            <div style="font-size: 2rem;">🧮</div>
            <div style="font-weight: 600; margin: 0.25rem 0;">Data Pipelines</div>
            <div class="neutral-subtext" style="font-size: 0.95rem;">Google Play + App Store + D2C data cleaned, normalized, and unified.</div>
        </div>
        <div class="neutral-card">
            <div style="font-size: 2rem;">🤖</div>
            <div style="font-weight: 600; margin: 0.25rem 0;">LLM Insights</div>
            <div class="neutral-subtext" style="font-size: 0.95rem;">Gemini generates insight JSON with confidence scoring and recommendations.</div>
        </div>
        <div class="neutral-card">
            <div style="font-size: 2rem;">📊</div>
            <div style="font-weight: 600; margin: 0.25rem 0;">Executive Reporting</div>
            <div class="neutral-subtext" style="font-size: 0.95rem;">Automated Markdown/HTML reports summarizing pipeline and insights.</div>
        </div>
        <div class="neutral-card">
            <div style="font-size: 2rem;">⚡</div>
            <div style="font-weight: 600; margin: 0.25rem 0;">Interactive UI</div>
            <div class="neutral-subtext" style="font-size: 0.95rem;">Minimal Streamlit workflow to inspect data, generate insights, and export deliverables.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Deliverables Overview
    st.markdown("### 📋 Assignment Deliverables")
    
    deliverables = [
        {
            "number": "1",
            "title": "Clean Combined Dataset",
            "description": "Unified CSV/JSON joining Google Play data with App Store samples.",
            "status": "Ready",
            "action": "Open Data Pipeline",
            "page": "📊 Data Pipeline Results"
        },
        {
            "number": "2",
            "title": "AI Insight JSON",
            "description": "Gemini-generated insights with confidence scoring and recommendations.",
            "status": "Ready (Download)",
            "action": "Download from Summary",
            "page": "📦 Deliverables Summary"
        },
        {
            "number": "3",
            "title": "Executive Report",
            "description": "Automated Markdown/HTML summary of data and insights.",
            "status": "Ready",
            "action": "Open Reports",
            "page": "📈 Executive Reports"
        },
        {
            "number": "4",
            "title": "Interactive Interface",
            "description": "Generate fresh insights via the Streamlit workflow and download JSON.",
            "status": "Ready",
            "action": "Open Live Generator",
            "page": "🤖 Live AI Generator"
        },
        {
            "number": "5",
            "title": "Phase 5 Extension",
            "description": "D2C funnel metrics, SEO opportunities & creative outputs (JSON).",
            "status": "Ready (Download)",
            "action": "Download from Summary",
            "page": "📦 Deliverables Summary"
        }
    ]

    st.markdown("""
    <div style="display: grid; gap: 1rem;">
    """, unsafe_allow_html=True)

    for deliverable in deliverables:
        st.markdown(f"""
        <div class="neutral-card" style="display: flex; flex-direction: column; gap: 0.5rem;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.25rem; font-weight: 600; color: #2563eb;">#{deliverable['number']}</span>
                    <span style="font-weight: 600;">{deliverable['title']}</span>
                </div>
                <span class="neutral-subtext" style="font-size: 0.85rem; font-weight: 500; color: #059669;">{deliverable['status']}</span>
            </div>
            <p class="neutral-subtext" style="margin: 0; font-size: 0.95rem;">{deliverable['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button(deliverable['action'], key=f"deliverable_action_{deliverable['number']}", use_container_width=True):
            st.session_state.page = deliverable['page']
            st.rerun()

    st.markdown("""
    </div>
    """, unsafe_allow_html=True)

    # Technical Implementation
    st.markdown("### 🏗️ Technical Implementation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="display: flex; flex-direction: column; gap: 0.75rem;">
            <div style="border: 1px solid #e2e8f0; border-radius: 0.75rem; padding: 1rem; background: var(--background-color, #fff);">
                <div style="font-weight: 600;">🚀 Data Pipeline</div>
                <ul style="color: #64748b; margin: 0.25rem 0 0 1.1rem;">
                    <li>Sources: Google Play (Kaggle), App Store API, Synthetic D2C.</li>
                    <li>Cleaning & normalization handled via dedicated pipeline modules.</li>
                    <li>Unified schema supports cross-platform comparisons.</li>
                </ul>
            </div>
            <div style="border: 1px solid #e2e8f0; border-radius: 0.75rem; padding: 1rem; background: var(--background-color, #fff);">
                <div style="font-weight: 600;">🤖 LLM Integration</div>
                <ul style="color: #64748b; margin: 0.25rem 0 0 1.1rem;">
                    <li>Google Gemini 2.5 Flash Lite with structured JSON prompts.</li>
                    <li>Confidence scores and evidence captured for every insight.</li>
                    <li>Reusable generator module for app + D2C analyses.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="display: flex; flex-direction: column; gap: 0.75rem;">
            <div style="border: 1px solid #e2e8f0; border-radius: 0.75rem; padding: 1rem; background: var(--background-color, #fff);">
                <div style="font-weight: 600;">🔗 API Integration</div>
                <ul style="color: #64748b; margin: 0.25rem 0 0 1.1rem;">
                    <li>RapidAPI App Store client with rate limiting & exponential backoff.</li>
                    <li>Exports review samples into the unified dataset for LLM prompts.</li>
                </ul>
            </div>
            <div style="border: 1px solid #e2e8f0; border-radius: 0.75rem; padding: 1rem; background: var(--background-color, #fff);">
                <div style="font-weight: 600;">📊 Output Generation</div>
                <ul style="color: #64748b; margin: 0.25rem 0 0 1.1rem;">
                    <li>Automated executive report (Markdown/HTML) via Jinja2 templates.</li>
                    <li>Deliverable-focused Streamlit UI for exploration and file exports.</li>
                    <li>All outputs stored under <code>data/processed/</code> and <code>reports/</code>.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start guide
    st.markdown("### 📚 Quick Start")
    st.markdown("""
    <ol style="color: #475569;">
        <li><strong>Explore data:</strong> Open <code>📊 Data Pipeline Results</code> to review the cleaned datasets.</li>
        <li><strong>Check reporting:</strong> Use <code>📈 Executive Reports</code> to regenerate or download the latest report.</li>
        <li><strong>Generate insights:</strong> Visit <code>🤖 Live AI Generator</code> to run Gemini on real-time reviews and export JSON.</li>
        <li><strong>Download deliverables:</strong> Go to <code>📦 Deliverables Summary</code> for all submission-ready files.</li>
    </ol>
    """, unsafe_allow_html=True)


def show_data_pipeline_results(gp_data, unified_data, d2c_data):
    """Show data pipeline results - Deliverable #1."""
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
                border-radius: 1rem; color: white; margin-bottom: 2rem;">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">📊 Data Pipeline Results</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem;">Deliverable #1: Clean Combined Dataset (CSV/JSON)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Pipeline overview
    st.markdown("### 🛠️ Pipeline Summary")
    
    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem;">
        <div class='neutral-card'>
            <div style="font-size: 1.75rem;">📊</div>
            <div style="font-weight: 600; margin: 0.25rem 0;">Google Play Store</div>
            <p class='neutral-subtext' style="font-size: 0.95rem;">Kaggle dataset (~10K apps) cleaned for missing values, normalized schema.</p>
        </div>
        <div class='neutral-card'>
            <div style="font-size: 1.75rem;">🍎</div>
            <div style="font-weight: 600; margin: 0.25rem 0;">App Store API</div>
            <p class='neutral-subtext' style="font-size: 0.95rem;">RapidAPI client with retries + rate limiting; reviews sampled for unified dataset.</p>
        </div>
        <div class='neutral-card'>
            <div style="font-size: 1.75rem;">🎯</div>
            <div style="font-weight: 600; margin: 0.25rem 0;">D2C Campaigns</div>
            <p class='neutral-subtext' style="font-size: 0.95rem;">Synthetic funnel dataset enriched with CAC, ROAS, retention, and SEO metrics.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-top: 1rem;">
    """, unsafe_allow_html=True)

    if gp_data is not None:
        st.metric("Google Play Apps", f"{len(gp_data):,}")
    if unified_data is not None:
        st.metric("App Store Samples", f"{len(unified_data[unified_data['source'] == 'app_store']):,}")
    if d2c_data is not None:
        st.metric("D2C Campaigns", f"{len(d2c_data):,}")

    st.markdown("""
    </div>
    """, unsafe_allow_html=True)
    
    # Unified dataset results
    st.markdown("### 🔄 Unified Dataset (Primary Deliverable)")
    
    if unified_data is not None:
        st.markdown("""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem;">
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Unified Rows", f"{len(unified_data):,}")
        with col2:
            gp_count = len(unified_data[unified_data['source'] == 'google_play'])
            st.metric("Google Play Rows", f"{gp_count:,}")
        with col3:
            as_count = len(unified_data[unified_data['source'] == 'app_store'])
            st.metric("App Store Rows", f"{as_count:,}")
        with col4:
            avg_rating = unified_data['rating'].mean()
            st.metric("Mean Rating", f"{avg_rating:.2f}")

        st.markdown("""
        </div>
        """, unsafe_allow_html=True)
        
        # Schema information
        st.markdown("### 📋 Unified Schema")
        st.markdown("""
        <div style="border: 1px solid #e2e8f0; border-radius: 0.75rem; padding: 1rem; background: var(--background-color, #fff);">
            <div style="font-weight: 600; color: #1f2937; margin-bottom: 0.25rem;">Unified Schema Highlights</div>
            <ul style="color: #64748b; margin: 0; padding-left: 1.1rem;">
                <li>Consistent columns: <code>app_name</code>, <code>category</code>, <code>rating</code>, <code>reviews</code>, <code>price</code>, <code>source</code>.</li>
                <li>Maintains platform-specific metadata (e.g., <code>last_updated</code>, <code>content_rating</code>).</li>
                <li>Ready for downstream analysis, LLM prompts, and reporting.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Data preview
        st.markdown("### 🔍 Quick Preview")

        with st.expander("View sample rows", expanded=False):
            sources = st.multiselect(
                "Filter by source:",
                unified_data['source'].unique(),
                default=unified_data['source'].unique()
            )
            filtered_data = unified_data[unified_data['source'].isin(sources)]
            st.dataframe(filtered_data.head(15), use_container_width=True)
        
        # Download section
        st.markdown("### 📥 Download Clean Dataset")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = unified_data.to_csv(index=False)
            st.download_button(
                label="📄 Download Unified CSV",
                data=csv_data,
                file_name="unified_data_clean.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            if gp_data is not None:
                gp_csv = gp_data.to_csv(index=False)
                st.download_button(
                    label="📱 Download Google Play CSV",
                    data=gp_csv,
                    file_name="google_play_cleaned.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col3:
            if d2c_data is not None:
                d2c_csv = d2c_data.to_csv(index=False)
                st.download_button(
                    label="🎯 Download D2C CSV",
                    data=d2c_csv,
                    file_name="d2c_campaigns_cleaned.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    else:
        st.error("🚀 Data not loaded. Please check the data pipeline.")


def show_executive_reports():
    """Show executive reports - Deliverable #3."""
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #059669 0%, #10b981 100%);
                border-radius: 1rem; color: white; margin-bottom: 2rem;">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">📈 Executive Reports</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem;">Deliverable #3: Automated Markdown/PDF/HTML Report Generation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Report overview
    st.markdown("### 📄 Generated Reports")
    
    # Check if reports exist - look for both HTML and Markdown
    html_report_path = "reports/executive_report.html"
    md_report_path = None
    
    # Look for generated markdown reports
    from pathlib import Path
    reports_dir = Path("reports")
    if reports_dir.exists():
        md_files = list(reports_dir.glob("executive_summary_*.md"))
        if md_files:
            # Get the most recent markdown file
            md_report_path = max(md_files, key=lambda x: x.stat().st_mtime)
    
    report_path = html_report_path if os.path.exists(html_report_path) else md_report_path
    
    if report_path and (report_path.exists() if isinstance(report_path, Path) else os.path.exists(str(report_path))):
        st.success("✅ Executive report successfully generated!")
        
        # Report details
        col1, col2, col3 = st.columns(3)
        
        with col1:
            format_type = "HTML" if str(report_path).endswith('.html') else "Markdown"
            st.metric("Report Format", format_type)
        with col2:
            # Get file size
            file_size = report_path.stat().st_size / 1024 if isinstance(report_path, Path) else os.path.getsize(report_path) / 1024
            st.metric("File Size", f"{file_size:.1f} KB")
        with col3:
            # Get modification time
            mod_time = datetime.fromtimestamp(report_path.stat().st_mtime if isinstance(report_path, Path) else os.path.getmtime(report_path))
            st.metric("Generated", mod_time.strftime("%Y-%m-%d %H:%M"))
        
        # Report contents preview
        st.markdown("### 🔍 Report Contents")
        
        st.info("""
        **📋 Executive Report includes:**
        - **Data Pipeline Summary:** Processing statistics and data quality metrics
        - **AI Insights Overview:** Key findings from LLM analysis with confidence scores
        - **App Store Intelligence:** Trends, features, and sentiment analysis
        - **D2C Campaign Analysis:** ROAS, funnel metrics, and optimization recommendations
        - **Technical Implementation:** Architecture, tools, and methodologies used
        - **Recommendations:** Actionable next steps based on analysis
        """)
        
        # Download section
        st.markdown("### 📥 Download Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Read and provide download
            with open(report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
            
            # Determine file type and download parameters
            if str(report_path).endswith('.html'):
                download_label = "📄 Download Executive Report (HTML)"
                file_name = "ai_market_intelligence_executive_report.html"
                mime_type = "text/html"
            else:
                download_label = "📄 Download Executive Report (Markdown)"
                file_name = "ai_market_intelligence_executive_report.md"
                mime_type = "text/markdown"
            
            st.download_button(
                label=download_label,
                data=report_content,
                file_name=file_name,
                mime=mime_type,
                use_container_width=True
            )
        
        with col2:
            if st.button("🔄 Regenerate Report", use_container_width=True):
                try:
                    # Import and run report generation
                    sys.path.append('src')
                    from reporting.generator import ReportGenerator
                    
                    with st.spinner("Regenerating executive report..."):
                        # Load the required data
                        import pandas as pd
                        from pathlib import Path
                        
                        # Load data for report generation
                        gp_data = pd.read_csv("data/processed/googleplaystore_cleaned.csv")
                        d2c_data = pd.read_csv("data/processed/d2c_cleaned.csv")
                        
                        # Load insights
                        insights = {}
                        insights_dir = Path("data/processed")
                        insight_files = list(insights_dir.glob("*insights*.json"))
                        for file_path in insight_files:
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                            if 'insights' in data:
                                insights[file_path.stem] = data['insights']
                            else:
                                insights[file_path.stem] = data
                        
                        generator = ReportGenerator()
                        report_path = generator.generate_executive_summary(gp_data, d2c_data, insights)
                        success = True
                        
                    if success:
                        st.success("✅ Report regenerated successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to regenerate report")
                except Exception as e:
                    st.error(f"❌ Error regenerating report: {e}")
    
    else:
        st.warning("🚀 Executive report not found. Let's generate it!")
        
        st.markdown("""
        **📈 About Executive Reports:**
        - Automatically generated using **Jinja2 templates**
        - Includes **data visualizations** and **key insights**
        - Professional **HTML format** ready for presentation
        - Contains **confidence scores** and **actionable recommendations**
        """)
        
        if st.button("🚀 Generate Executive Report", type="primary", use_container_width=True):
            try:
                sys.path.append('src')
                from reporting.generator import ReportGenerator
                
                with st.spinner("Generating executive report..."):
                    # Load the required data
                    import pandas as pd
                    from pathlib import Path
                    
                    # Load data for report generation
                    gp_data = pd.read_csv("data/processed/googleplaystore_cleaned.csv")
                    d2c_data = pd.read_csv("data/processed/d2c_cleaned.csv")
                    
                    # Load insights
                    insights = {}
                    insights_dir = Path("data/processed")
                    insight_files = list(insights_dir.glob("*insights*.json"))
                    for file_path in insight_files:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        if 'insights' in data:
                            insights[file_path.stem] = data['insights']
                        else:
                            insights[file_path.stem] = data
                    
                    generator = ReportGenerator()
                    report_path = generator.generate_executive_summary(gp_data, d2c_data, insights)
                    success = True
                
                if success:
                    st.success("✅ Executive report generated successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to generate report")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.info("📝 Make sure the report generator is properly configured.")


def show_live_ai_generator():
    """Show live AI generator - Deliverable #4: CLI/Streamlit interface to query insights."""
    
    # Header with clean design
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
                border-radius: 1rem; color: white; margin-bottom: 2rem;">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">🤖 Interactive Query Interface</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem;">Deliverable #4: CLI/Streamlit Interface to Query Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("""
    **🎯 Purpose:** This interface demonstrates the "CLI/Streamlit interface to query insights" requirement.
    
    **What it does:**
    - Query real-time app store data via API
    - Generate AI insights with confidence scores
    - Export structured JSON files (Deliverable #2)
    - Save results for submission
    
    **Generated files become Deliverable #2:** The JSON outputs from this interface are the required "Insights JSON file (with confidence scores & recommendations)"
    """)

    def render_app_result(result: dict):
        """Render the most recent App Store AI insight result and export options."""
        if not result or 'insights' not in result:
            return

        insights = result['insights']
        selected_app_name = result.get('app_name', 'Selected App')
        total_reviews_used = result.get('total_reviews_used', insights.get('total_reviews_analyzed', 0))
        avg_rating_used = result.get('average_rating', result.get('avg_rating_used', insights.get('average_rating', 0.0)))
        pages_used = result.get('pages_used')
        page_size = result.get('page_size')
        status_message = result.get('status_message')
        fetch_message = result.get('fetch_message')

        if fetch_message:
            st.info(fetch_message)
        if status_message:
            st.success(status_message)

        st.markdown("""
        <div style="text-align: center; padding: 0.75rem; background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    border-radius: 0.75rem; color: white; margin: 1rem 0;">
            <h3 style="margin: 0; font-size: 1.25rem;">✅ Analysis Complete!</h3>
            <p style="margin: 0.25rem 0 0 0; font-size: 0.9rem;">{0} reviews analyzed for {1}</p>
        </div>
        """.format(total_reviews_used, selected_app_name), unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Reviews Analyzed", insights.get('total_reviews_analyzed', total_reviews_used))

        with col2:
            st.metric("Average Rating", f"{insights.get('average_rating', avg_rating_used):.2f}/5")

        with col3:
            confidence = max([f.get('confidence_score', 0) for f in insights.get('requested_features', [])] + [0])
            st.metric("Analysis Confidence", f"{confidence:.0%}")

        with st.expander("💭 Sentiment Analysis", expanded=True):
            st.write(insights.get('sentiment_summary', 'No sentiment analysis available'))

        features = insights.get('requested_features', [])
        if features:
            with st.expander("🏆 Top Features", expanded=False):
                for i, feature in enumerate(features[:3], 1):
                    with st.container():
                        st.markdown(f"**{i}. {feature.get('feature', 'N/A')}**")
                        st.caption(f"Confidence: {feature.get('confidence_score', 0):.1%}")
                        st.write(feature.get('recommendation', 'No recommendation'))

        complaints = insights.get('common_complaints', [])
        if complaints:
            with st.expander("⚠️ Key Issues", expanded=False):
                for i, complaint in enumerate(complaints[:3], 1):
                    with st.container():
                        st.markdown(f"**{i}. {complaint.get('complaint', 'N/A')}**")
                        st.caption(f"Confidence: {complaint.get('confidence_score', 0):.1%}")
                        st.write(complaint.get('recommendation', 'No recommendation'))

        st.markdown("### 💾 Export Results")

        col1, col2, col3 = st.columns([1, 1, 1])

        insights_json = json.dumps(insights, indent=2, ensure_ascii=False)

        with col1:
            st.download_button(
                label="📄 Download JSON",
                data=insights_json,
                file_name=result.get('download_filename', 'ai_insights.json'),
                mime="application/json",
                help="Download complete analysis as JSON",
                width="stretch",
                key=f"download_live_ai_app_json_{result.get('download_filename', 'ai_insights.json')}_{datetime.now().timestamp()}"
            )

        with col2:
            if st.button("💾 Save to Project", key="save_live_ai_app", help="Save to project data folder", use_container_width=True):
                try:
                    os.makedirs("data/processed", exist_ok=True)
                    slug = result.get('file_slug', re.sub(r'[^a-z0-9]+', '_', selected_app_name.lower()).strip('_') or 'app_insight')
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"ai_insights_{slug}_{timestamp}.json"
                    filepath = Path("data/processed") / filename

                    metadata = {
                        "generated_at": result.get('generated_at', datetime.now().isoformat()),
                        "model_used": result.get('model_used', 'gemini-2.5-flash-lite'),
                        "generator_version": "1.0",
                        "source": "streamlit_live_ai_generator",
                        "app_id": result.get('app_id'),
                        "app_name": selected_app_name,
                        "reviews_requested": result.get('target_reviews'),
                        "reviews_deduplicated": result.get('deduped_count'),
                        "reviews_analyzed": total_reviews_used,
                        "average_rating": avg_rating_used,
                        "pages_fetched": pages_used,
                        "page_size_estimate": page_size,
                        "filter_english_only": result.get('filter_english_only', True),
                        "sort_option": result.get('sort_option', 'mostRecent')
                    }

                    payload = {
                        "metadata": metadata,
                        "insights": insights
                    }

                    with filepath.open('w', encoding='utf-8') as f:
                        json.dump(payload, f, indent=2, ensure_ascii=False)

                    st.session_state.live_ai_app_last_saved = str(filepath)
                    st.success(f"✅ Saved to {filepath}!")

                except Exception as e:
                    st.error(f"Save error: {e}")

        with col3:
            if st.button("🗑️ Clear Result", key="clear_live_ai_app", use_container_width=True):
                st.session_state.pop('live_ai_app_result', None)
                st.session_state.pop('live_ai_app_last_saved', None)
                st.success("Cleared generated insight.")
                st.session_state.live_ai_app_cleared = True

        if st.session_state.get('live_ai_app_last_saved'):
            st.caption(f"Last saved to: {st.session_state['live_ai_app_last_saved']}")

    def render_d2c_result(result: dict):
        """Render the most recent D2C insights result."""
        if not result or 'insights' not in result:
            return

        insights = result['insights']
        seo_category = result.get('seo_category', insights.get('seo_category', 'D2C Category'))
        summary = result.get('summary', {})
        status_message = result.get('status_message')

        if status_message:
            st.success(status_message)

        total_campaigns = insights.get('total_campaigns', summary.get('campaigns', 0))
        avg_roas = summary.get('avg_roas')
        avg_ctr = summary.get('avg_ctr')
        total_revenue = summary.get('total_revenue')

        st.markdown("""
        <div style="text-align: center; padding: 0.75rem; background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
                    border-radius: 0.75rem; color: white; margin: 1rem 0;">
            <h3 style="margin: 0; font-size: 1.25rem;">✅ D2C Analysis Complete!</h3>
            <p style="margin: 0.25rem 0 0 0; font-size: 0.9rem;">{0} campaigns analyzed for {1}</p>
        </div>
        """.format(total_campaigns, seo_category), unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Campaigns", total_campaigns)
        with col2:
            if avg_roas is not None:
                st.metric("Average ROAS", f"{avg_roas:.2f}x")
        with col3:
            if avg_ctr is not None:
                st.metric("Average CTR", f"{avg_ctr:.2f}%")
        with col4:
            if total_revenue is not None:
                st.metric("Total Revenue", f"${total_revenue:,.0f}")

        top_performers = insights.get('top_performers', [])
        if top_performers:
            with st.expander("🏆 Top Performing Campaigns", expanded=True):
                for performer in top_performers[:3]:
                    name = performer.get('campaign_name', 'Campaign')
                    roas = performer.get('roas', 0)
                    conv = performer.get('conversion_rate', 0)
                    st.markdown(f"**{name}** — ROAS {roas:.2f}x | Conversion {conv:.2f}%")
                    st.write(performer.get('success_factors', 'No success factors provided.'))

        optimizations = insights.get('optimization_recommendations', [])
        if optimizations:
            with st.expander("🛠️ Optimization Recommendations", expanded=False):
                for rec in optimizations:
                    st.markdown(f"**{rec.get('campaign_type', 'Campaign Type')}**")
                    st.caption(rec.get('current_performance', ''))
                    st.write(rec.get('recommended_actions', ''))

        creative = insights.get('creative_content', {})
        if creative:
            with st.expander("🎨 Creative Outputs", expanded=False):
                headlines = creative.get('ad_headlines', [])
                if headlines:
                    st.write("**Ad Headlines:**")
                    for idx, headline in enumerate(headlines, 1):
                        st.write(f"{idx}. {headline}")
                if creative.get('meta_description'):
                    st.write(f"**Meta Description:** {creative['meta_description']}")

        strategic = insights.get('strategic_insights', {})
        if strategic:
            with st.expander("🧭 Strategic Insights", expanded=False):
                if strategic.get('overall_health'):
                    st.write(f"**Overall Health:** {strategic['overall_health']}")
                if strategic.get('opportunities'):
                    st.write("**Opportunities:**")
                    for item in strategic['opportunities']:
                        st.write(f"- {item}")
                if strategic.get('recommendations'):
                    st.write("**Recommendations:**")
                    for item in strategic['recommendations']:
                        st.write(f"- {item}")

        st.markdown("### 💾 Export Results")
        col1, col2, col3 = st.columns(3)
        export_json = json.dumps(insights, indent=2, ensure_ascii=False)

        with col1:
            st.download_button(
                label="📄 Download JSON",
                data=export_json,
                file_name=result.get('download_filename', 'd2c_insights.json'),
                mime="application/json",
                help="Download D2C insight JSON",
                width="stretch",
                key=f"download_live_ai_d2c_json_{result.get('download_filename', 'd2c_insights.json')}_{datetime.now().timestamp()}"
            )

        with col2:
            if st.button("💾 Save to Project", key="save_live_ai_d2c", use_container_width=True):
                try:
                    os.makedirs("data/processed", exist_ok=True)
                    slug = result.get('file_slug', re.sub(r'[^a-z0-9]+', '_', seo_category.lower()).strip('_') or 'd2c_insight')
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filepath = Path("data/processed") / f"d2c_insights_{slug}_{timestamp}.json"

                    metadata = {
                        "generated_at": result.get('generated_at', datetime.now().isoformat()),
                        "model_used": result.get('model_used', 'gemini-2.5-flash-lite'),
                        "generator_version": "1.0",
                        "source": "streamlit_live_ai_generator_d2c",
                        "seo_category": seo_category,
                        "campaigns_provided": summary.get('campaigns'),
                        "average_roas": summary.get('avg_roas'),
                        "average_ctr": summary.get('avg_ctr'),
                        "total_revenue": summary.get('total_revenue'),
                        "channels_included": summary.get('channels')
                    }

                    payload = {"metadata": metadata, "insights": insights}
                    with filepath.open('w', encoding='utf-8') as f:
                        json.dump(payload, f, indent=2, ensure_ascii=False)

                    st.session_state.live_ai_d2c_last_saved = str(filepath)
                    st.success(f"✅ Saved to {filepath}!")
                except Exception as exc:
                    st.error(f"Save error: {exc}")

        with col3:
            if st.button("🗑️ Clear Result", key="clear_live_ai_d2c", use_container_width=True):
                st.session_state.pop('live_ai_d2c_result', None)
                st.session_state.pop('live_ai_d2c_last_saved', None)
                st.success("Cleared generated insight.")
                st.session_state.live_ai_d2c_cleared = True

        if st.session_state.get('live_ai_d2c_last_saved'):
            st.caption(f"Last saved to: {st.session_state['live_ai_d2c_last_saved']}")

    @st.cache_data(show_spinner=False)
    def load_d2c_campaigns() -> pd.DataFrame:
        try:
            return pd.read_csv("data/processed/d2c_cleaned.csv")
        except FileNotFoundError:
            st.error("❌ D2C dataset not found. Run the data pipeline first.")
            return pd.DataFrame()
        except Exception as exc:
            st.error(f"❌ Error loading D2C dataset: {exc}")
            return pd.DataFrame()

    try:
        generator = GeminiInsightsGenerator()
        app_store_client = AppStoreClient()

        st.success("✅ System ready for analysis!")

        tab_apps, tab_d2c = st.tabs(["📱 App Store Analysis", "🎯 D2C Campaign Analysis"])

        with tab_apps:
            if st.session_state.pop('live_ai_app_cleared', False):
                st.experimental_rerun()

            if st.session_state.get('live_ai_app_result'):
                st.markdown("---")
                st.markdown("### 🔁 Most Recent App Insight")
                render_app_result(st.session_state['live_ai_app_result'])
                st.markdown("---")

            st.markdown("### 🔍 Select App")
            selected_app_id: Optional[str] = None
            col_select, col_manual = st.columns([2, 1])

            with col_select:
                popular_apps = {
                    "📚 Apple Books": "364709193",
                    "📷 Instagram": "389801252",
                    "💬 WhatsApp": "310633997",
                    "🎥 YouTube": "544007664",
                    "🎵 TikTok": "835599320"
                }

                app_display = st.selectbox(
                    "Choose Popular App:",
                    list(popular_apps.keys()),
                    key="live_ai_app_select",
                    help="Select a common app or provide a custom ID"
                )
                if app_display:
                    selected_app_id = popular_apps[app_display]

            with col_manual:
                manual_app_id = st.text_input(
                    "Custom App ID:",
                    placeholder="e.g., 544007664",
                    key="live_ai_app_manual",
                    help="Enter any App Store ID"
                )
                if manual_app_id.strip().isdigit():
                    selected_app_id = manual_app_id.strip()

            st.markdown("### ⚙️ Analysis Settings")
            st.markdown("Using fixed data source: **US App Store** reviews in **English** for consistency.")

            col_sort, col_reviews = st.columns(2)
            with col_sort:
                sort_option = st.selectbox(
                    "📊 Sort By",
                    ["Most Recent", "Most Helpful", "Newest"],
                    index=0,
                    key="live_ai_app_sort"
                )

            with col_reviews:
                max_reviews = st.slider(
                    "📝 Reviews to Analyze",
                    min_value=10,
                    max_value=500,
                    step=10,
                    value=50,
                    key="live_ai_app_reviews",
                    help="RapidAPI provides 50 reviews per page (up to 10 pages)"
                )

            with st.expander("🔧 Advanced Options"):
                filter_english_only = st.checkbox(
                    "English Reviews Only",
                    value=True,
                    key="live_ai_app_filter_english",
                    help="Disable to include multilingual reviews"
                )
                st.caption("💡 Leave enabled for deliverable consistency")

            if st.button("🚀 Generate AI Insights", type="primary", use_container_width=True, key="live_ai_generate_app"):
                if not selected_app_id:
                    st.error("❌ Please select an app or enter an App Store ID")
                else:
                    app_name = None
                    for display_name, app_id in popular_apps.items():
                        if app_id == selected_app_id:
                            app_name = display_name.split(' ', 1)[1]
                            break
                    if not app_name:
                        app_name = f"App {selected_app_id}"

                    status_placeholder = st.empty()
                    status_placeholder.info("📡 Fetching reviews...")

                    with st.spinner("🤖 Analyzing app reviews with AI..."):
                        try:
                            sort_map = {
                                "Most Recent": "mostRecent",
                                "Most Helpful": "mostHelpful",
                                "Newest": "newest"
                            }
                            api_sort = sort_map.get(sort_option, "mostRecent")

                            progress_bar = st.progress(0)

                            def update_status(current_page: int):
                                progress_bar.progress(min(0.2 + (current_page - 1) * 0.1, 0.9))
                                status_placeholder.info(f"📡 Fetching page {current_page} of App Store reviews...")

                            target_reviews = min(max_reviews, 500)
                            raw_reviews, _, pages_used, page_size = fetch_reviews_for_app(
                                app_store_client,
                                selected_app_id,
                                country="us",
                                sort=api_sort,
                                lang="en",
                                target_count=target_reviews,
                                status_callback=update_status,
                                start_page=1,
                                max_pages=10
                            )

                            if not raw_reviews:
                                st.error("❌ Failed to fetch reviews. Please try again later or choose another app.")
                                return

                            progress_bar.progress(0.9)

                            deduped_reviews: list[dict] = []
                            seen_ids = set()
                            for review in raw_reviews:
                                review_id = review.get('id') or review.get('reviewId') or (
                                    f"{review.get('userName', '')}_{review.get('updated', '')}"
                                )
                                if review_id in seen_ids:
                                    continue
                                seen_ids.add(review_id)
                                deduped_reviews.append(review)

                            status_placeholder.info(
                                f"📦 Retrieved {len(deduped_reviews)} unique reviews across {pages_used} page(s)"
                            )

                            filtered_reviews = list(deduped_reviews)
                            if filter_english_only:
                                def is_english(text: str) -> bool:
                                    if not text:
                                        return False
                                    latin_chars = len(re.findall(r'[a-zA-Z]', text))
                                    total_chars = len(re.sub(r"\s+", '', text))
                                    if total_chars == 0:
                                        return False
                                    latin_ratio = latin_chars / total_chars
                                    common_words = {
                                        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                                        'is', 'are', 'was', 'were', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                                        'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
                                        'app', 'good', 'bad', 'great', 'love', 'hate', 'like', 'nice', 'awesome', 'terrible',
                                        'amazing', 'perfect', 'worst', 'best'
                                    }
                                    text_lower = text.lower()
                                    english_word_count = sum(1 for word in common_words if word in text_lower)
                                    return latin_ratio > 0.7 and (english_word_count > 0 or len(text.split()) < 3)

                                original_count = len(filtered_reviews)
                                filtered_reviews = [r for r in filtered_reviews if is_english(r.get('text', ''))]
                                filtered_count = len(filtered_reviews)
                                if filtered_count < original_count:
                                    st.info(f"📝 Filtered {original_count - filtered_count} non-English reviews (kept {filtered_count})")

                            reviews = filtered_reviews[:target_reviews]
                            if not reviews:
                                st.error("❌ No reviews available after filtering. Adjust options and try again.")
                                return

                            def compute_average_rating(items: list[dict]) -> float:
                                ratings = []
                                for item in items:
                                    rating_val = item.get('score', item.get('rating'))
                                    if rating_val is None:
                                        continue
                                    try:
                                        ratings.append(float(rating_val))
                                    except (TypeError, ValueError):
                                        continue
                                return sum(ratings) / len(ratings) if ratings else 0.0

                            total_reviews_used = len(reviews)
                            avg_rating_used = round(compute_average_rating(reviews), 2)

                            progress_bar.progress(1.0)
                            status_placeholder.success(
                                f"✅ {total_reviews_used} reviews ready (pages fetched: {pages_used}, page size ≈ {page_size})"
                            )

                            status_placeholder.info("🧠 Generating AI insights...")
                            category_name = f"{app_name}_ANALYSIS"
                            insights = generator.generate_insights(reviews, category_name)

                            if insights:
                                insights['total_reviews_analyzed'] = total_reviews_used
                                insights['average_rating'] = avg_rating_used

                                result_payload = {
                                    "insights": insights,
                                    "app_id": selected_app_id,
                                    "app_name": app_name,
                                    "total_reviews_used": total_reviews_used,
                                    "average_rating": avg_rating_used,
                                    "target_reviews": target_reviews,
                                    "deduped_count": len(deduped_reviews),
                                    "pages_used": pages_used,
                                    "page_size": page_size,
                                    "filter_english_only": filter_english_only,
                                    "sort_option": sort_option,
                                    "download_filename": f"ai_insights_{app_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    "file_slug": re.sub(r'[^a-z0-9]+', '_', app_name.lower()).strip('_') or 'app_insight',
                                    "status_message": "✅ Analysis complete!",
                                    "fetch_message": f"📦 Retrieved {len(deduped_reviews)} unique reviews across {pages_used} page(s)",
                                    "generated_at": datetime.now().isoformat(),
                                    "model_used": "gemini-2.5-flash-lite"
                                }
                                st.session_state.live_ai_app_result = result_payload
                                st.session_state.live_ai_app_last_saved = None
                                st.markdown("### Latest App Insight")
                                render_app_result(result_payload)
                            else:
                                st.error("❌ Gemini did not return valid JSON. Please try again.")

                        except Exception as exc:
                            st.error(f"❌ Error during insights generation: {exc}")
                            st.info("💡 Possible API limit or network hiccup. Try again shortly.")

        with tab_d2c:
            if st.session_state.pop('live_ai_d2c_cleared', False):
                st.experimental_rerun()

            d2c_df = load_d2c_campaigns()
            if d2c_df.empty:
                st.warning("Load the D2C dataset via the data pipeline to enable this tab.")
            else:
                if st.session_state.get('live_ai_d2c_result'):
                    st.markdown("---")
                    st.markdown("### 🔁 Most Recent D2C Insight")
                    render_d2c_result(st.session_state['live_ai_d2c_result'])
                    st.markdown("---")

                st.markdown("### 🎯 Select D2C Segment")
                categories = sorted(d2c_df['seo_category'].unique())
                selected_category = st.selectbox(
                    "SEO Category",
                    categories,
                    key="live_ai_d2c_category",
                    help="Choose a funnel/SEO category"
                )

                category_df = d2c_df[d2c_df['seo_category'] == selected_category].copy()
                campaign_count = len(category_df)

                summary_stats = {
                    "campaigns": campaign_count,
                    "avg_roas": float(category_df['roas'].mean()) if campaign_count else None,
                    "avg_ctr": float(category_df['ctr'].mean()) if campaign_count else None,
                    "total_revenue": float(category_df['revenue_usd'].sum()) if campaign_count else None,
                    "channels": int(category_df['channel'].nunique()) if campaign_count else None
                }

                col_c1, col_c2, col_c3, col_c4 = st.columns(4)
                with col_c1:
                    st.metric("Campaigns", campaign_count)
                with col_c2:
                    st.metric("Avg ROAS", f"{summary_stats['avg_roas']:.2f}x" if summary_stats['avg_roas'] is not None else "-")
                with col_c3:
                    st.metric("Avg CTR", f"{summary_stats['avg_ctr']:.2f}%" if summary_stats['avg_ctr'] is not None else "-")
                with col_c4:
                    st.metric("Channels", summary_stats['channels'] if summary_stats['channels'] is not None else "-")

                with st.expander("🔍 Preview Campaign Data", expanded=False):
                    st.dataframe(category_df.head(10), use_container_width=True)

                if st.button("🚀 Generate D2C Insights", type="primary", use_container_width=True, key="live_ai_generate_d2c"):
                    if campaign_count == 0:
                        st.error("❌ No campaigns available for this category.")
                    else:
                        with st.spinner("🤖 Generating D2C insights with AI..."):
                            try:
                                insights = generator.generate_d2c_insights(category_df, selected_category)
                                if insights:
                                    result_payload = {
                                        "insights": insights,
                                        "seo_category": selected_category,
                                        "summary": summary_stats,
                                        "download_filename": f"d2c_insights_{re.sub(r'[^a-z0-9]+', '_', selected_category.lower()).strip('_') or 'category'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                        "file_slug": re.sub(r'[^a-z0-9]+', '_', selected_category.lower()).strip('_') or 'd2c_insight',
                                        "generated_at": datetime.now().isoformat(),
                                        "model_used": "gemini-2.5-flash-lite",
                                        "status_message": f"✅ Analyzed {campaign_count} campaigns across {summary_stats['channels'] or 0} channels"
                                    }
                                    st.session_state.live_ai_d2c_result = result_payload
                                    st.session_state.live_ai_d2c_last_saved = None
                                    st.markdown("### Latest D2C Insight")
                                    render_d2c_result(result_payload)
                                else:
                                    st.error("❌ Gemini did not return valid D2C JSON. Please try again.")
                            except Exception as exc:
                                st.error(f"❌ Error generating D2C insights: {exc}")

    except Exception as e:
        st.error(f"❌ Error initializing AI system: {e}")
        st.info("💡 Make sure your API keys are properly configured in the .env file")


def show_deliverables_summary(gp_data, unified_data, d2c_data, insights):
    """Show deliverables summary - Complete submission package overview."""
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
                border-radius: 1rem; color: white; margin-bottom: 2rem;">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">📦 Deliverables Summary</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem;">Complete Submission Package - All Required Files & Downloads</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Submission checklist
    st.markdown("### ✅ Assignment Completion Checklist")
    
    deliverables_status = [
        {
            "id": "1",
            "title": "Clean Combined Dataset (CSV/JSON)",
            "file": "unified_data.csv",
            "status": "✅" if unified_data is not None else "❌",
            "description": f"Unified dataset with {len(unified_data):,} records" if unified_data is not None else "Not available"
        },
        {
            "id": "2",
            "title": "Insights JSON (confidence scores & recommendations)", 
            "file": "ai_insights_*.json",
            "status": "✅" if insights else "❌",
            "description": f"{len(insights)} insight files generated" if insights else "No insights generated"
        },
        {
            "id": "3",
            "title": "Executive Report (Markdown/PDF/HTML)",
            "file": "executive_report.*",
            "status": "✅" if (os.path.exists("reports/executive_report.html") or (os.path.exists("reports") and any(f.endswith('.md') and 'executive_summary' in f for f in os.listdir("reports")))) else "❌",
            "description": "Automated report with analysis" if (os.path.exists("reports/executive_report.html") or (os.path.exists("reports") and any(f.endswith('.md') and 'executive_summary' in f for f in os.listdir("reports")))) else "Report not generated"
        },
        {
            "id": "4",
            "title": "CLI/Streamlit Interface",
            "file": "app.py",
            "status": "✅",
            "description": "Interactive interface for querying insights (this application)"
        },
        {
            "id": "5",
            "title": "Phase 5 Extension: D2C + SEO + Creative",
            "file": "d2c_insights_*.json",
            "status": "✅" if any('d2c' in k for k in insights.keys()) else "❌",
            "description": "Funnel insights, SEO analysis, and AI-generated creative content"
        }
    ]
    
    for deliverable in deliverables_status:
        col1, col2, col3, col4 = st.columns([1, 3, 2, 3])
        
        with col1:
            st.markdown(f"**#{deliverable['id']}** {deliverable['status']}")
        with col2:
            st.markdown(f"**{deliverable['title']}**")
        with col3:
            st.code(deliverable['file'])
        with col4:
            st.write(deliverable['description'])
    
    # Download section
    st.markdown("### 📥 Download Deliverables")

    sections = [
        {
            "title": "📊 Primary Data Files",
            "content": lambda: download_data_files(unified_data, d2c_data)
        },
        {
            "title": "🧠 AI Insights (JSON)",
            "content": lambda: download_insights_files(insights)
        },
        {
            "title": "📈 Reports & Documentation",
            "content": lambda: download_report_and_docs()
        }
    ]

    for section in sections:
        with st.expander(section["title"], expanded=True):
            section["content"]()
    
    # System metrics
    st.markdown("### 📊 System Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_records = len(unified_data) if unified_data is not None else 0
        st.metric("Total Data Records", f"{total_records:,}")
    
    with col2:
        total_insights = len(insights) if insights else 0
        st.metric("AI Insights Generated", total_insights)
    
    with col3:
        total_campaigns = len(d2c_data) if d2c_data is not None else 0
        st.metric("D2C Campaigns Analyzed", total_campaigns)
    
    with col4:
        # Count total features analyzed
        total_features = 0
        if insights:
            for insight_data in insights.values():
                data = insight_data if 'insights' not in insight_data else insight_data['insights']
                if 'requested_features' in data:
                    total_features += len(data['requested_features'])
        st.metric("Features Identified", total_features)
    
    # Submission instructions
    st.markdown("### 📫 Submission Instructions")
    
    st.info("""
    **📦 Complete Submission Package:**
    
    1. **Download all files** using the buttons above
    2. **Include source code** - the entire `ai-market-intelligence/` project folder
    3. **Add a demo video** showing the Streamlit interface in action
    4. **Verify requirements.txt** includes all dependencies
    
    **🎬 Demo Suggestions:**
    - Navigate through each deliverable page
    - Generate a new insight using the Live AI Generator
    - Show the data pipeline results and downloads
    - Demonstrate the executive report
    
    **📝 Ready for Evaluation:**
    All 5 deliverables are complete and demonstrable through this interface.
    """)


if __name__ == "__main__":
    main()
