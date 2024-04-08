from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from pathlib import Path
import yaml
# from src.utils import Fetcher, Manager, CACHE
# from src.model import unet

app = Flask(__name__)
CORS(app)

# # Load configuration
# with open(Path(__file__).parent / "src/backend_model.yaml", 'r') as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)

@app.route('/inference/tissue/', methods=['GET'])
def inference_tissue():
    """
    Mocked endpoint for tissue segmentation.
    Expected query parameters: series (required)
    """
    args = request.args  # Query parameters from the request
    series = args.get('series')

    if not series:
        return jsonify({"error": "Series ID is required."}), 400

    try:
        # Enhanced mocked response with DICOM header information
        mocked_response = {
            "series": 1,
            "patientInfo": {
                "patientID": "KCL_0001",
                "patientName": "John Doe",
                "patientBirthDate": "19700101",
                "patientSex": "M",
                "studyDate": "20221031",
                "modality": "CT",
                "manufacturer": "Unknown manufacturer",
                "modelName": "Unknown model"
            },
            "imageInfo": {
                "position": "[-108, 132, -110]",
                "orientation": "[1, 0, 0, 0, -1, 0]",
                "windowCenter": 338.47,
                "windowWidth": 2724.94
            },
            "volume": {
                "total": {
                    "wm": 500.0,
                    "gm": 300.0,
                    "csf": 200.0
                },
                "display": {
                    "wm": "/path/to/wm_image_url",
                    "gm": "/path/to/gm_image_url",
                    "csf": "/path/to/csf_image_url"
                }
            },
            "message": "This is a mocked response including DICOM header information"
        }

        return jsonify(mocked_response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
