import subprocess
import sys
import os
import argparse
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_backend():
    print("Starting FastAPI Backend (backend/main.py)...")
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
        cwd=ROOT_DIR
    )

def run_training(steps):
    print(f"Starting RL Training ({steps} steps)...")
    subprocess.run(
        [sys.executable, "train_v2.py", "--total-steps", str(steps)],
        cwd=os.path.join(ROOT_DIR, "RLModel")
    )

def main():
    parser = argparse.ArgumentParser(description="QuantPilot Unified Runner")
    parser.add_argument("--train", type=int, help="Run training for N steps")
    parser.add_argument("--serve", action="store_true", help="Launch Backend Server")
    args = parser.parse_args()

    if args.train:
        run_training(args.train)
    
    if args.serve:
        backend_proc = run_backend()
        
        print("\nBackend is LIVE!")
        print("   - API Docs: http://localhost:8000/docs")
        print("   - Health:   http://localhost:8000/health")
        print("   - WS Stream: ws://localhost:8000/api/v1/stream")
        print("\nTo run the Flutter UI, open a new terminal and run:")
        print("   cd rl_trading_ui && flutter run")
        print("\nPress Ctrl+C to stop the server.")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down server...")
            backend_proc.terminate()
    elif not args.train:
        parser.print_help()

if __name__ == "__main__":
    main()
