#deletes all pyc files
find . -name "*.pyc" -print0 | xargs -0 rm -rf
find . -name "__pycache__" -print0 | xargs -0 rm -rf
