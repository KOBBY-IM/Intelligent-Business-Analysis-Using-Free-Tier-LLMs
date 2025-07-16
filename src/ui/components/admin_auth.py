"""
Inline Admin Authentication Component

Provides reusable inline admin login functionality for pages that require admin access.
"""

import streamlit as st
import os
import time
from dotenv import load_dotenv

load_dotenv()

def show_inline_admin_login(page_title: str = "Admin Access Required") -> bool:
    """
    Display inline admin login form and handle authentication.
    
    Args:
        page_title: Title to display for the admin section
        
    Returns:
        bool: True if authenticated, False if not (calling page should return/stop)
    """
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "changeme")
    
    # Rate limiting for login attempts
    if 'login_attempts' not in st.session_state:
        st.session_state['login_attempts'] = 0
    if 'last_attempt_time' not in st.session_state:
        st.session_state['last_attempt_time'] = 0
    
    # Check if already authenticated
    if st.session_state.get('admin_logged_in'):
        return True
    
    # Apply styling for admin login section
    st.markdown("""
    <style>
    .admin-login-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin: 2rem 0;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .admin-login-title {
        color: white;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .admin-login-subtitle {
        color: rgba(255,255,255,0.9);
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main container
    st.markdown(f"""
    <div class="admin-login-container">
        <h2 class="admin-login-title">🔐 {page_title}</h2>
        <p class="admin-login-subtitle">Administrator credentials required to access this page</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Security warning for default password
    if ADMIN_PASSWORD == "changeme":
        st.error("⚠️ **SECURITY WARNING**: Default admin password detected! Please set ADMIN_PASSWORD environment variable.")
    
    # Rate limiting check
    current_time = time.time()
    if st.session_state['login_attempts'] >= 3 and (current_time - st.session_state['last_attempt_time']) < 300:
        remaining_time = 300 - (current_time - st.session_state['last_attempt_time'])
        st.error(f"🔒 Too many failed attempts. Please wait {int(remaining_time)} seconds before trying again.")
        return False
    
    # Create centered login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("inline_admin_login"):
            st.markdown("### 🔑 Enter Admin Password")
            password = st.text_input(
                "Admin Password", 
                type="password", 
                placeholder="Enter admin password",
                help="Contact the system administrator if you don't have the password"
            )
            
            submit = st.form_submit_button(
                "🚀 Access Admin Dashboard", 
                type="primary", 
                use_container_width=True
            )
            
            if submit:
                if password == ADMIN_PASSWORD:
                    st.session_state['admin_logged_in'] = True
                    st.session_state['login_attempts'] = 0  # Reset attempts on successful login
                    st.success("✅ Admin authentication successful!")
                    st.balloons()
                    st.rerun()  # Refresh to show the protected content
                else:
                    st.session_state['login_attempts'] += 1
                    st.session_state['last_attempt_time'] = current_time
                    st.error(f"❌ Incorrect password. Attempt {st.session_state['login_attempts']}/3")
                    
                    if st.session_state['login_attempts'] >= 3:
                        st.warning("🔒 Account will be locked for 5 minutes after 3 failed attempts.")
    
    # Show additional info
    st.info("💡 **Admin access provides:** Advanced analytics, data export, batch testing, and system management capabilities.")
    
    return False


def show_admin_header(page_name: str):
    """Display admin header with logout option."""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"### 👋 Welcome, Administrator")
        st.caption(f"Currently viewing: **{page_name}**")
    
    with col3:
        if st.button("🚪 Logout", help="Logout from admin dashboard"):
            st.session_state['admin_logged_in'] = False
            st.success("Logged out successfully!")
            st.rerun()
    
    st.markdown("---") 