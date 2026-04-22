import os
import logging
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Prometheus metrics
alerts_received = Counter('alerts_received_total', 'Total alerts received', ['severity'])
webhook_processing_time = Histogram('webhook_processing_seconds', 'Time to process webhook')

@app.route('/webhook', methods=['POST'])
@webhook_processing_time.time()
def handle_alert():
    """Handle alerts from AlertManager"""
    try:
        data = request.json
        
        # Check if data is None
        if data is None:
            logger.warning("Received empty webhook payload")
            return jsonify({'error': 'Empty payload'}), 400
        
        logger.info(f"Received alert: {data}")
        
        # Count alerts by severity
        alerts = data.get('alerts', [])
        if not alerts:
            logger.warning("No alerts in payload")
            return jsonify({'status': 'ok', 'alerts_count': 0}), 200
        
        for alert in alerts:
            severity = alert.get('labels', {}).get('severity', 'unknown')
            alerts_received.labels(severity=severity).inc()
            logger.info(f"Processed alert with severity: {severity}")
        
        return jsonify({'status': 'ok', 'alerts_count': len(alerts)}), 200
    except TypeError as e:
        logger.error(f"Type error processing alert: {e}")
        return jsonify({'error': 'Invalid payload format'}), 400
    except Exception as e:
        logger.error(f"Error processing alert: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'service': 'MLOps Auto-Trigger',
        'version': '1.0.0',
        'endpoints': {
            '/webhook': 'POST - Receive alerts from AlertManager',
            '/health': 'GET - Health check',
            '/metrics': 'GET - Prometheus metrics',
            '/': 'GET - This endpoint'
        }
    }), 200

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    logger.warning(f"Route not found: {request.path}")
    return jsonify({'error': 'Not found', 'path': request.path}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting MLOps Auto-Trigger service on {host}:{port}")
    app.run(host=host, port=port, debug=debug)