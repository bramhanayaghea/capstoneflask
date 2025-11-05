# ============================================================================
# app.py - FLASK WEB APPLICATION
# ============================================================================

from flask import Flask, render_template, request, jsonify
import model
import sys

app = Flask(__name__)

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """Handle recommendation requests"""
    try:
        # Get username from form
        username = request.form.get('username', '').strip()
        
        if not username:
            return jsonify({
                'success': False,
                'error': 'Please enter a username'
            })
        
        # Get recommendations
        recommendations, error = model.recommend_top_5_products(username)
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            })
        
        # Convert DataFrame to list of dictionaries
        recommendations_list = recommendations.to_dict('records')
        
        return jsonify({
            'success': True,
            'username': username,
            'recommendations': recommendations_list
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


