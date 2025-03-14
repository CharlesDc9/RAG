import subprocess
import sys
from pathlib import Path
import time
import webbrowser

def run_services():
    # Get the project root directory
    root_dir = Path(__file__).parent
    
    try:
        print("Starting FastAPI backend...")
        backend = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--reload"],
            cwd=root_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print("Starting Streamlit frontend...")
        frontend = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", 
             str(root_dir / "app" / "frontend" / "streamlit_app.py"),
             "--server.port=8501",
             "--browser.serverAddress=localhost",
             "--server.headless=true",  # Skip email prompt
             "--browser.gatherUsageStats=false"],  # Disable usage stats
            cwd=root_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a bit for services to start
        time.sleep(2)
        
        print("\nServices started successfully!")
        print("Backend running at: http://localhost:8000")
        print("Frontend running at: http://localhost:8501")
        
        # Try to open the browser
        try:
            webbrowser.open("http://localhost:8501")
        except:
            print("Could not open browser automatically. Please navigate to http://localhost:8501")
        
        # Keep the services running and show their output
        while True:
            # Print backend output
            backend_out = backend.stdout.readline()
            if backend_out:
                print(f"Backend: {backend_out.decode()}", end='')
                
            # Print frontend output
            frontend_out = frontend.stdout.readline()
            if frontend_out:
                print(f"Frontend: {frontend_out.decode()}", end='')
                
            # Check if either process has ended
            if backend.poll() is not None or frontend.poll() is not None:
                print("One of the services stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\nShutting down services...")
        backend.terminate()
        frontend.terminate()
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        try:
            backend.terminate()
            frontend.terminate()
        except:
            pass
        sys.exit(1)

if __name__ == "__main__":
    run_services()