import time
from residue import core

def verify_isolation_tiers():
    print("--- Verifying V4.1 Isolation Zone ---")
    
    # We create the controller which instantiates ResidueWall -> IsolationZone
    # Actually IsolationZone is used per worker_loop in AsyncObserver, 
    # but the simplest way to print the system static report is directly:
    core.print_isolation_report()
    
    print("\nStarting AsyncObserver to trigger IsolationZone lock...")
    observer = core.AsyncObserver(1024, 100)
    observer.start()
    time.sleep(0.5)
    
    telem = observer.poll_telemetry()
    print(f"Background Isolation Active: {telem.isolation_active}")
    
    observer.stop()
    print("--- Verification Complete ---")

if __name__ == "__main__":
    verify_isolation_tiers()
