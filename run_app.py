import subprocess
import webbrowser
import time
import os
import sys

def run_streamlit_app():
    """Launch Streamlit app and automatically open browser"""
    
    print("🚀 Starting Streamlit Dashboard...")
    print("📊 Dashboard: Analisis Wisatawan Nusantara")
    print("=" * 50)
    
    # Check if app.py exists
    if not os.path.exists('app.py'):
        print("❌ Error: app.py tidak ditemukan!")
        print("Pastikan file app.py berada di direktori yang sama")
        return
    
    # Check if dataset.csv exists
    if not os.path.exists('dataset.csv'):
        print("❌ Error: dataset.csv tidak ditemukan!")
        print("Pastikan file dataset.csv berada di direktori yang sama")
        return
    
    try:
        # Launch Streamlit with specific port and auto-open browser
        cmd = [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ]
        
        print("🌐 Opening browser in 3 seconds...")
        time.sleep(3)
        
        # Open browser automatically
        webbrowser.open('http://localhost:8501')
        
        # Start Streamlit process
        print("✅ Streamlit dashboard berhasil dibuka!")
        print("🔗 URL: http://localhost:8501")
        print("⏹️  Tekan Ctrl+C untuk menghentikan server")
        print("=" * 50)
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard dihentikan oleh user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Coba jalankan manual dengan: streamlit run app.py")

if __name__ == "__main__":
    run_streamlit_app() 