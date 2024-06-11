from flask import Flask, request, redirect, url_for, render_template
from models.submissionclass import *  # Import the FileContainer class

import os

file_container = FileContainer()
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        files = request.files.getlist('file')
        uploaded_files = {'ply': None, 'png': None, 'json': None}

        # Save uploaded files and organize them by extension
        for file in files:
            if file:
                filename = file.filename
                extension = os.path.splitext(filename)[1].lower()[1:]
                if extension in uploaded_files:
                    uploaded_files[extension] = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(uploaded_files[extension])

        # Ensure all required files are present before creating FileContainer
        if None not in uploaded_files.values():
            file_container.loaddata(
                uploaded_files['ply'],
                uploaded_files['png'],  # Prefer jpg if both jpg and png are present
                uploaded_files['json']
            )
            imagetoshow, _ = file_container.detectscrews()  # Get the image from detectscrews
            file_container.annotateallscrews()
            displaypointclouds(file_container.annotated_point_cloud)

            return render_template('display.html', image=imagetoshow)  # Pass image to the template


        # Handle error scenario where not all required files are uploaded
        else:
            return "Error: Please upload all required files (PLY, JPG/PNG, JSON)."


        return render_template('display.html', image=imagetoshow)  # Pass image to the template

@app.route('/display')
def display_images():
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('display.html', images=images)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
