import urllib.request, json, time, threading
import uvicorn

def run():
    uvicorn.run('app.api.main:app', host='127.0.0.1', port=8899, log_level='warning')

def test_endpoints():
    print('PROJECT CHRONOS — FINAL ENDPOINT VERIFICATION')
    print('=' * 50)
    
    endpoints = [
        ('/', 'Root'),
        ('/api/v1/system/health', 'Health'),
        ('/api/v1/patients', 'Patients List'),
        ('/api/v1/alerts', 'Alerts'),
        ('/api/v1/analytics/alarm-fatigue', 'Alarm Fatigue'),
        ('/api/v1/analytics/validation', 'Validation'),
        ('/api/v1/analytics/dashboard-summary', 'Dashboard Summary')
    ]
    
    for path, name in endpoints:
        try:
            req = urllib.request.urlopen(f'http://127.0.0.1:8899{path}', timeout=5)
            data = json.loads(req.read())
            print(f'✅ {name:20} → {len(str(data))} bytes')
        except Exception as e:
            print(f'❌ {name:20} → ERROR: {str(e)}')
    
    print('=' * 50)
    print('ENDPOINT VERIFICATION COMPLETE')

# Start server in background
import subprocess
subprocess.Popen(['python', '-c', 'import uvicorn; uvicorn.run("app.api.main:app", host="127.0.0.1", port=8899, log_level="warning")'], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Wait for server to start
time.sleep(8)

# Test endpoints
test_endpoints()
