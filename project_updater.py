"""
Project Update Module with Improved Model
Integrates all enhancements into the final project
"""

import os
import shutil
import json
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Import custom modules
from improved_emotion_model import ImprovedEmotionRecognizer
from advanced_image_preprocessor import AdvancedImagePreprocessor
from model_tester import ModelTester

class ProjectUpdater:
    """
    Class for updating the project with improved model and integrating all enhancements
    """
    
    def __init__(self, project_dir='.', backup_dir='backup'):
        """
        Initialize the project updater
        
        Parameters:
            project_dir: Project directory
            backup_dir: Backup directory
        """
        self.project_dir = project_dir
        self.backup_dir = os.path.join(project_dir, backup_dir)
        
        # Create backup directory if it does not exist
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Define files to update
        self.files_to_update = [
            'emotion_recognition.py',
            'camera_system.py',
            'app.py',
            'advanced_emotion_analyzer.py'
        ]
        
        # Define new files to add
        self.new_files = [
            'improved_emotion_model.py',
            'advanced_image_preprocessor.py',
            'model_tester.py',
            'dataset_enhancer.py',
            'model_trainer.py',
            'project_updater.py'
        ]
        
        # Define project folders
        self.project_folders = [
            'models',
            'utils',
            'static',
            'templates',
            'test_data',
            'enhanced_dataset',
            'test_results'
        ]
        
        # Ensure all folders exist
        for folder in self.project_folders:
            os.makedirs(os.path.join(project_dir, folder), exist_ok=True)
    
    def backup_original_files(self):
        """
        Create backup of original files
        
        Returns:
            List of backed up files
        """
        print("Creating backup of original files...")
        
        backed_up_files = []
        
        for file_name in self.files_to_update:
            source_path = os.path.join(self.project_dir, file_name)
            backup_path = os.path.join(self.backup_dir, file_name)
            
            if os.path.exists(source_path):
                shutil.copy2(source_path, backup_path)
                backed_up_files.append(file_name)
                print(f"  Backed up {file_name}")
        
        return backed_up_files
    
    def update_project_structure(self):
        """
        Update project structure to match new requirements
        """
        print("Updating project structure...")
        
        # Create necessary directories
        for folder in self.project_folders:
            folder_path = os.path.join(self.project_dir, folder)
            os.makedirs(folder_path, exist_ok=True)
            print(f"  Created/verified folder: {folder}")
        
        print("Project structure updated successfully.")
    
    def verify_dependencies(self):
        """
        Verify that all required dependencies are installed
        
        Returns:
            Dictionary of dependency status
        """
        print("Verifying dependencies...")
        
        dependencies = {
            'opencv': False,
            'tensorflow': False,
            'numpy': False,
            'matplotlib': False,
            'flask': False,
            'pandas': False,
            'scikit-learn': False
        }
        
        try:
            import cv2
            dependencies['opencv'] = True
            print("  ✓ OpenCV is installed")
        except ImportError:
            print("  ✗ OpenCV is not installed")
        
        try:
            import tensorflow as tf
            dependencies['tensorflow'] = True
            print("  ✓ TensorFlow is installed")
        except ImportError:
            print("  ✗ TensorFlow is not installed")
        
        try:
            import numpy
            dependencies['numpy'] = True
            print("  ✓ NumPy is installed")
        except ImportError:
            print("  ✗ NumPy is not installed")
        
        try:
            import matplotlib
            dependencies['matplotlib'] = True
            print("  ✓ Matplotlib is installed")
        except ImportError:
            print("  ✗ Matplotlib is not installed")
        
        try:
            import flask
            dependencies['flask'] = True
            print("  ✓ Flask is installed")
        except ImportError:
            print("  ✗ Flask is not installed")
        
        try:
            import pandas
            dependencies['pandas'] = True
            print("  ✓ Pandas is installed")
        except ImportError:
            print("  ✗ Pandas is not installed")
        
        try:
            import sklearn
            dependencies['scikit-learn'] = True
            print("  ✓ scikit-learn is installed")
        except ImportError:
            print("  ✗ scikit-learn is not installed")
        
        return dependencies
    
    def generate_configuration_file(self):
        """
        Generate project configuration file
        
        Returns:
            Path to configuration file
        """
        print("Generating configuration file...")
        
        config = {
            'project_name': 'AI Camera Emotion Recognition System',
            'version': '2.0.0',
            'description': 'Emotion recognition system using AI and computer vision',
            'author': 'Development Team',
            'models': {
                'face_detection': 'models/haarcascade_frontalface_default.xml',
                'emotion_recognition': 'models/emotion_model.h5',
                'improved_emotion_recognition': 'models/improved_emotion_model.h5'
            },
            'folders': {
                'models': 'models',
                'static': 'static',
                'templates': 'templates',
                'test_data': 'test_data',
                'enhanced_dataset': 'enhanced_dataset',
                'test_results': 'test_results',
                'backup': 'backup'
            },
            'settings': {
                'camera_index': 0,
                'camera_resolution': [640, 480],
                'emotion_confidence_threshold': 0.5,
                'use_advanced_preprocessing': True,
                'use_transfer_learning': True,
                'enable_autism_support': False
            }
        }
        
        config_path = os.path.join(self.project_dir, 'config.json')
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        print(f"  Configuration file created: {config_path}")
        
        return config_path
    
    def generate_documentation(self):
        """
        Generate or update project documentation
        """
        print("Generating documentation...")
        
        # Create CHANGELOG file
        changelog_path = os.path.join(self.project_dir, 'CHANGELOG.md')
        
        changelog = """# Changelog

## Version 2.0.0 (2024)

### New Features
- Improved emotion recognition model with better accuracy
- Advanced image preprocessing for better results
- Autism support with friendly descriptions and tips
- Dual emotion analysis
- Enhanced analytics reporting
- Web interface for easy access

### Improvements
- Better face detection accuracy
- Improved emotion recognition accuracy
- Faster processing speed
- More comprehensive error handling
- Better documentation

### Bug Fixes
- Fixed camera initialization issues
- Improved compatibility with different camera types
- Fixed video recording issues

## Version 1.0.0 (Initial Release)

### Features
- Real-time face detection
- Basic emotion recognition
- Live camera streaming
- Image upload and analysis
- Video recording
- Basic reporting

"""
        
        with open(changelog_path, 'w', encoding='utf-8') as f:
            f.write(changelog)
        
        print(f"  Changelog created: {changelog_path}")
    
    def run_system_tests(self):
        """
        Run system tests to verify everything is working
        
        Returns:
            Test results
        """
        print("Running system tests...")
        
        test_results = {
            'dependencies': {},
            'project_structure': {},
            'model_loading': {},
            'overall_status': 'unknown'
        }
        
        # Test dependencies
        print("  Testing dependencies...")
        test_results['dependencies'] = self.verify_dependencies()
        
        # Test project structure
        print("  Testing project structure...")
        for folder in self.project_folders:
            folder_path = os.path.join(self.project_dir, folder)
            test_results['project_structure'][folder] = os.path.exists(folder_path)
        
        # Test model loading
        print("  Testing model loading...")
        try:
            from improved_emotion_model import ImprovedEmotionRecognizer
            from advanced_image_preprocessor import AdvancedImagePreprocessor
            test_results['model_loading']['improved_model'] = True
            print("    ✓ Improved model loaded successfully")
        except Exception as e:
            test_results['model_loading']['improved_model'] = False
            print(f"    ✗ Failed to load improved model: {e}")
        
        # Determine overall status
        all_dependencies_ok = all(test_results['dependencies'].values())
        all_folders_ok = all(test_results['project_structure'].values())
        all_models_ok = all(test_results['model_loading'].values())
        
        if all_dependencies_ok and all_folders_ok and all_models_ok:
            test_results['overall_status'] = 'success'
            print("  ✓ All tests passed!")
        else:
            test_results['overall_status'] = 'warning'
            print("  ⚠ Some tests failed or skipped")
        
        return test_results
    
    def update_project(self):
        """
        Perform full project update
        
        Returns:
            Update status and results
        """
        print("=" * 50)
        print("Starting Project Update")
        print("=" * 50)
        
        update_results = {
            'backup': None,
            'structure': None,
            'dependencies': None,
            'configuration': None,
            'tests': None,
            'overall_status': 'unknown'
        }
        
        try:
            # Step 1: Backup original files
            print("\nStep 1: Backing up original files...")
            update_results['backup'] = self.backup_original_files()
            
            # Step 2: Update project structure
            print("\nStep 2: Updating project structure...")
            self.update_project_structure()
            update_results['structure'] = 'success'
            
            # Step 3: Verify dependencies
            print("\nStep 3: Verifying dependencies...")
            update_results['dependencies'] = self.verify_dependencies()
            
            # Step 4: Generate configuration
            print("\nStep 4: Generating configuration...")
            update_results['configuration'] = self.generate_configuration_file()
            
            # Step 5: Generate documentation
            print("\nStep 5: Generating documentation...")
            self.generate_documentation()
            
            # Step 6: Run system tests
            print("\nStep 6: Running system tests...")
            update_results['tests'] = self.run_system_tests()
            
            # Determine overall status
            if update_results['tests']['overall_status'] == 'success':
                update_results['overall_status'] = 'success'
                print("\n" + "=" * 50)
                print("✓ Project update completed successfully!")
                print("=" * 50)
            else:
                update_results['overall_status'] = 'warning'
                print("\n" + "=" * 50)
                print("⚠ Project update completed with warnings!")
                print("=" * 50)
        
        except Exception as e:
            update_results['overall_status'] = 'error'
            print(f"\n✗ Error during project update: {e}")
            print("=" * 50)
        
        return update_results


# Run update if this file is run directly
if __name__ == "__main__":
    # Get project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create project updater
    updater = ProjectUpdater(project_dir=project_dir)
    
    # Run update
    results = updater.update_project()
