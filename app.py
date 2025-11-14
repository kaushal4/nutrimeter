# /3d-vision-api/app.py

from server import create_app

app = create_app()

if __name__ == '__main__':
    # We use app.config['DEBUG'] from config.py
    app.run(debug=app.config['DEBUG'], host='0.0.0.0', port=5000)