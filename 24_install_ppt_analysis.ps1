# This script sets up the Python virtual environment for the PPT analysis tool.

# 1. Create the virtual environment
python -m venv ppt_analysis_venv

# 2. Upgrade pip
./ppt_analysis_venv/Scripts/python.exe -m pip install --upgrade pip

# 3. Install dependencies
./ppt_analysis_venv/Scripts/python.exe -m pip install "pillow>=10.0" "tqdm>=4.66" pdf2image pypdf2 "langgraph>=0.0.21" "python-pptx>=0.6.23" "openai>=1.3.0" "pydantic>=1.10.0" "black>=23.7.0" "isort>=5.12.0" "flake8>=6.1.0" "pygame>=2.5" "pywin32>=306"

Write-Host "PPT analysis venv setup complete. Please activate the environment using '.\ppt_analysis_venv\Scripts\activate'."
