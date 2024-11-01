import requests

def test_base_url():
  response = requests.get('http://localhost:5000/')
  assert response.status_code == 200
  assert 'pdf_file' in response.text