from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str) -> list[str]:
    """
    This function returns the list of requirements
    """
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        # Clean each line
        requirements = [req.strip() for req in requirements]
        
        requirements = [req.replace("\n", "") for req in requirements if req != '-e .']
    
    return requirements 

setup(
    name='mlproject',
    version='0.0.1',
    author='ReihanBo',
    author_email='reihaneh.boustani@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    
)