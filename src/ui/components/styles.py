"""
Centralized styling module for the LLM Blind Evaluation System

This module provides consistent styling components and HTML templates
to ensure maintainable and consistent UI across the application.
"""

import streamlit as st
from typing import Optional, Dict, Any


class UIStyles:
    """Centralized styling constants and methods"""
    
    # Color palette
    COLORS = {
        'primary': '#1f77b4',
        'success': '#28a745',
        'warning': '#ffc107',
        'error': '#dc3545',
        'info': '#17a2b8',
        'secondary': '#6c757d',
        'light': '#f8f9fa',
        'dark': '#343a40',
        'text': '#201f1e',
        'text_secondary': '#666',
        'background': '#f3f2f1'
    }
    
    # Typography
    FONTS = {
        'primary': "'Segoe UI', 'Roboto', 'Inter', sans-serif",
        'mono': "'Consolas', 'Monaco', 'Courier New', monospace"
    }
    
    # Font sizes
    FONT_SIZES = {
        'xs': '1rem',
        'sm': '1.125rem',
        'md': '1.25rem',
        'lg': '1.375rem',
        'xl': '1.625rem',
        'xxl': '2rem',
        'xxxl': '2.5rem'
    }
    
    # Spacing
    SPACING = {
        'xs': '0.5rem',
        'sm': '1rem',
        'md': '1.5rem',
        'lg': '2rem',
        'xl': '3rem'
    }
    
    # Border radius
    RADIUS = {
        'sm': '4px',
        'md': '8px',
        'lg': '12px',
        'xl': '16px'
    }

    @staticmethod
    def get_base_styles() -> str:
        """Get base CSS styles for the application"""
        return f"""
        <style>
        /* Global Fluent Design System Styles */
        .stApp {{
            background-color: {UIStyles.COLORS['background']} !important;
            font-family: {UIStyles.FONTS['primary']} !important;
            color: {UIStyles.COLORS['text']} !important;
            font-size: {UIStyles.FONT_SIZES['md']} !important;
            line-height: 1.6 !important;
        }}
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {{
            color: {UIStyles.COLORS['text']} !important;
            font-family: {UIStyles.FONTS['primary']} !important;
            font-weight: 600 !important;
            margin-bottom: {UIStyles.SPACING['sm']} !important;
            line-height: 1.4 !important;
        }}
        
        h1 {{
            font-size: {UIStyles.FONT_SIZES['xxxl']} !important;
        }}
        
        h2 {{
            font-size: {UIStyles.FONT_SIZES['xxl']} !important;
        }}
        
        h3 {{
            font-size: {UIStyles.FONT_SIZES['xl']} !important;
        }}
        
        /* Text elements */
        p, div, span, label {{
            font-size: {UIStyles.FONT_SIZES['md']} !important;
            line-height: 1.6 !important;
        }}
        
        /* Streamlit specific elements */
        .stMarkdown {{
            font-size: {UIStyles.FONT_SIZES['md']} !important;
        }}
        
        .stButton > button {{
            font-size: {UIStyles.FONT_SIZES['md']} !important;
            font-weight: 500 !important;
        }}
        
        .stSelectbox label {{
            font-size: {UIStyles.FONT_SIZES['lg']} !important;
            font-weight: 500 !important;
        }}
        
        .stTextInput label {{
            font-size: {UIStyles.FONT_SIZES['xl']} !important;
            font-weight: 600 !important;
            color: {UIStyles.COLORS['text']} !important;
            margin-bottom: 0.75rem !important;
        }}
        
        .stCheckbox label {{
            font-size: {UIStyles.FONT_SIZES['xl']} !important;
            font-weight: 600 !important;
            color: {UIStyles.COLORS['text']} !important;
            margin-bottom: 0.5rem !important;
        }}
        
        .stTextArea label {{
            font-size: {UIStyles.FONT_SIZES['lg']} !important;
            font-weight: 500 !important;
        }}
        
        .stForm {{
            background: white;
            padding: {UIStyles.SPACING['md']};
            border-radius: {UIStyles.RADIUS['lg']};
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: {UIStyles.SPACING['sm']} 0;
        }}
        
        .stExpander {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: {UIStyles.RADIUS['md']};
            margin: {UIStyles.SPACING['sm']} 0;
        }}
        
        .stExpander > div > div {{
            padding: {UIStyles.SPACING['sm']};
        }}
        
        .stRadio label {{
            font-size: {UIStyles.FONT_SIZES['lg']} !important;
            font-weight: 500 !important;
        }}
        
        /* Card-like containers */
        .ui-card {{
            background: white;
            border-radius: {UIStyles.RADIUS['lg']};
            box-shadow: 0 2px 8px rgba(0,0,0,0.07), 0 1.5px 4px rgba(0,0,0,0.03);
            padding: {UIStyles.SPACING['md']};
            margin-bottom: {UIStyles.SPACING['sm']};
        }}
        
        /* Centered content */
        .ui-center {{
            text-align: center;
            padding: {UIStyles.SPACING['lg']} 0;
        }}
        
        /* Status indicators */
        .ui-success {{
            color: {UIStyles.COLORS['success']} !important;
        }}
        
        .ui-warning {{
            color: {UIStyles.COLORS['warning']} !important;
        }}
        
        .ui-error {{
            color: {UIStyles.COLORS['error']} !important;
        }}
        
        .ui-info {{
            color: {UIStyles.COLORS['info']} !important;
        }}
        
        /* Progress indicators */
        .ui-progress-item {{
            display: flex;
            align-items: center;
            margin-bottom: {UIStyles.SPACING['xs']};
        }}
        
        .ui-progress-icon {{
            margin-right: {UIStyles.SPACING['xs']};
        }}
        </style>
        """

    @staticmethod
    def render_header(title: str, subtitle: Optional[str] = None, icon: str = "", color: str = "primary") -> None:
        """Render a consistent header with optional subtitle and icon"""
        # Use pure Streamlit components instead of HTML
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            # Main title
            st.markdown(f"# {icon} {title}")
            
            # Subtitle if provided
            if subtitle:
                st.markdown(f"## {subtitle}")
            
            # Add some spacing
            st.markdown("")

    @staticmethod
    def render_card(content: str, title: Optional[str] = None) -> None:
        """Render content in a card-like container"""
        card_html = "<div class='ui-card'>"
        
        if title:
            card_html += f"<h3>{title}</h3>"
        
        card_html += f"{content}</div>"
        
        st.markdown(card_html, unsafe_allow_html=True)

    @staticmethod
    def render_progress_item(text: str, completed: bool = False) -> None:
        """Render a progress item with consistent styling"""
        icon = "✅" if completed else "⏳"
        class_name = "ui-success" if completed else "ui-info"
        
        st.markdown(
            f'<div class="ui-progress-item"><span class="ui-progress-icon">{icon}</span><span class="{class_name}">{text}</span></div>',
            unsafe_allow_html=True
        )

    @staticmethod
    def render_section_divider() -> None:
        """Render a consistent section divider"""
        st.markdown("---")

    @staticmethod
    def render_info_section(title: str, items: list, icon: str = "ℹ️") -> None:
        """Render an information section with bullet points"""
        st.markdown(f"### {icon} {title}")
        for item in items:
            st.markdown(f"- {item}")

    @staticmethod
    def apply_base_styles() -> None:
        """Apply base styles to the current page"""
        st.markdown(UIStyles.get_base_styles(), unsafe_allow_html=True)


class PageTemplates:
    """Pre-built page templates for common layouts"""
    
    @staticmethod
    def render_completion_page(
        user_summary: Optional[Dict[str, Any]] = None,
        show_restart_button: bool = True
    ) -> None:
        """Render the completion/thank you page"""
        
        # Apply base styles
        UIStyles.apply_base_styles()
        
        # Header
        UIStyles.render_header(
            title="Evaluation Complete!",
            subtitle="Thank you for participating in our research",
            icon="✅",
            color="success"
        )
        
        UIStyles.render_section_divider()
        
        # Summary section
        if user_summary:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("## 📊 Your Evaluation Summary")
                
                # Display completion status
                st.markdown("**Industries Completed:**")
                retail_completed = user_summary.get('retail_completed', False)
                finance_completed = user_summary.get('finance_completed', False)
                
                retail_icon = "✅" if retail_completed else "⏳"
                finance_icon = "✅" if finance_completed else "⏳"
                
                st.markdown(f"{retail_icon} **🛍️ Retail**: {'All 6 questions completed' if retail_completed else 'In progress'}")
                st.markdown(f"{finance_icon} **💰 Finance**: {'All 6 questions completed' if finance_completed else 'In progress'}")
                
                UIStyles.render_section_divider()
        
        # Thank you section
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("## 🎉 Thank You!")
            
            st.markdown("""
            Your participation in this research study is greatly appreciated. 
            Your feedback will help us understand how different AI models perform 
            in real-world business scenarios across multiple industries.
            """)
            
            UIStyles.render_info_section(
                title="What Happens Next?",
                items=[
                    "Your responses have been securely saved",
                    "The research team will analyze the data",
                    "Results will be published in academic format",
                    "No personal information will be shared"
                ],
                icon="📋"
            )
            
            UIStyles.render_info_section(
                title="Research Impact",
                items=[
                    "Understanding LLM performance in business contexts",
                    "Improving AI decision support systems",
                    "Advancing responsible AI development",
                    "Cross-industry AI evaluation methodologies"
                ],
                icon="🔬"
            )
            
            UIStyles.render_section_divider()
            
            # Restart button
            if show_restart_button:
                if st.button("🏠 Start New Evaluation", type="primary", use_container_width=True):
                    # Reset all session state
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()

    @staticmethod
    def render_sidebar_progress(current_step: int, session_id: Optional[str] = None) -> None:
        """Render consistent sidebar progress indicator"""
        st.sidebar.title("🔬 Blind Evaluation")
        UIStyles.render_section_divider()
        
        if st.session_state.get('registered'):
            st.sidebar.markdown("**Progress:**")
            progress_steps = ["Introduction", "Evaluation", "Complete"]
            
            for i, step in enumerate(progress_steps):
                UIStyles.render_progress_item(step, completed=(i <= current_step))
            
            # Show current industry if in evaluation
            if current_step == 1:
                current_industry = st.session_state.get("current_industry", "retail")
                retail_progress = st.session_state.get("retail_question_idx", 0)
                finance_progress = st.session_state.get("finance_question_idx", 0)
                
                st.sidebar.markdown(f"**Current:** {current_industry.title()}")
                
                # Show industry progress
                st.sidebar.markdown("**Industry Progress:**")
                st.sidebar.markdown(f"🛍️ Retail: {retail_progress}/6")
                st.sidebar.markdown(f"💰 Finance: {finance_progress}/6")
        
        UIStyles.render_section_divider()
        
        # Session info
        if session_id:
            st.sidebar.markdown(f"**Session ID:** {session_id[:8]}...")


# Convenience functions for backward compatibility
def render_header(title: str, subtitle: Optional[str] = None, icon: str = "", color: str = "primary") -> None:
    """Convenience function for rendering headers"""
    UIStyles.render_header(title, subtitle, icon, color)

def render_completion_page(user_summary: Optional[Dict[str, Any]] = None, show_restart_button: bool = True) -> None:
    """Convenience function for rendering completion page"""
    PageTemplates.render_completion_page(user_summary, show_restart_button)

def render_sidebar_progress(current_step: int, session_id: Optional[str] = None) -> None:
    """Convenience function for rendering sidebar progress"""
    PageTemplates.render_sidebar_progress(current_step, session_id)

def apply_base_styles() -> None:
    """Convenience function for applying base styles"""
    UIStyles.apply_base_styles() 