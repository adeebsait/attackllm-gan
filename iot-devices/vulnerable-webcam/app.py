from flask import Flask, request, jsonify
import subprocess
import base64

app = Flask(__name__)

# Hardcoded credentials for the 'login'
ADMIN_USER = "admin"
ADMIN_PASS = "admin"


@app.route('/')
def index():
    return "Vulnerable IoT Webcam Interface. Use /login to authenticate.", 200


@app.route('/login')
def login():
    auth = request.authorization
    if not auth or auth.username != ADMIN_USER or auth.password != ADMIN_PASS:
        return 'Authentication required.', 401, {'WWW-Authenticate': 'Basic realm="Login Required"'}

    # In a real app, you'd return a token. Here, we'll just confirm success.
    # We'll encode a success message in base64 to simulate a simple token.
    token = base64.b64encode(b"logged_in_successfully").decode('utf-8')
    return jsonify({"status": "login successful", "token": token}), 200


@app.route('/exec')
def execute_command():
    # --- VULNERABILITY 1: Authentication Check ---
    # Check for the simple 'logged in' status from the /login route.
    # In a real scenario, this would be a proper token check.
    auth = request.authorization
    if not auth or auth.username != ADMIN_USER or auth.password != ADMIN_PASS:
        return 'Authentication required.', 401, {'WWW-Authenticate': 'Basic realm="Login Required"'}

    # --- VULNERABILITY 2: Command Injection ---
    # Get command from the 'cmd' query parameter and execute it.
    # This is highly insecure and is the intended vulnerability.
    cmd = request.args.get('cmd')
    if not cmd:
        return jsonify({"error": "No command provided. Use ?cmd=... parameter."}), 400

    try:
        # Using shell=True to demonstrate the vulnerability clearly
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        return jsonify({"output": result}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Command failed to execute", "output": e.output}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

