#!/usr/bin/env python3
"""
XRoboToolkit Teleop Data Log Analysis Script

This script analyzes the structure of .pkl files in the logs directory to understand
the teleoperation data format, including robot state data and camera images.

Usage:
    python test_data_log_analysis.py [pkl_file_path]
    
    If no file path is provided, it will analyze the first .pkl file found in logs/.
"""

import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np


class TeleopDataAnalyzer:
    """Analyzes teleoperation pickle log files to understand their structure."""
    
    def __init__(self, pkl_file_path: str):
        """
        Initialize the analyzer with a pickle file path.
        
        Args:
            pkl_file_path (str): Path to the pickle file to analyze
        """
        self.pkl_file_path = Path(pkl_file_path)
        self.data: Optional[List[Dict]] = None
        self.analysis_results: Dict[str, Any] = {}
    
    def load_data(self) -> bool:
        """
        Safely load the pickle file data.
        
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            print(f"Loading pickle file: {self.pkl_file_path}")
            print(f"File size: {self.pkl_file_path.stat().st_size / (1024*1024):.2f} MB")
            
            with open(self.pkl_file_path, 'rb') as f:
                self.data = pickle.load(f)
                
            print(f"✅ Successfully loaded {len(self.data)} data entries")
            return True
            
        except Exception as e:
            print(f"❌ Error loading pickle file: {e}")
            return False
    
    def get_data_type_info(self, value: Any) -> str:
        """
        Get detailed type information for a value.
        
        Args:
            value: The value to analyze
            
        Returns:
            str: Descriptive type information
        """
        if isinstance(value, np.ndarray):
            return f"numpy.ndarray(shape={value.shape}, dtype={value.dtype})"
        elif isinstance(value, list):
            if len(value) == 0:
                return "list(empty)"
            elif len(value) < 10:
                return f"list(length={len(value)}, types={[type(x).__name__ for x in value]})"
            else:
                return f"list(length={len(value)}, sample_types={[type(value[i]).__name__ for i in [0, len(value)//2, -1]]})"
        elif isinstance(value, dict):
            return f"dict(keys={list(value.keys())[:5]}{'...' if len(value) > 5 else ''})"
        elif isinstance(value, (int, float)):
            return f"{type(value).__name__}({value})"
        elif isinstance(value, str):
            return f"str(length={len(value)}, preview='{value[:50]}{'...' if len(value) > 50 else ''}')"
        else:
            return f"{type(value).__name__}"
    
    def analyze_data_structure(self):
        """Analyze the overall structure of the loaded data."""
        if not self.data:
            print("❌ No data loaded")
            return
        
        print("\n" + "="*80)
        print("DATA STRUCTURE ANALYSIS")
        print("="*80)
        
        # Basic info
        self.analysis_results['total_entries'] = len(self.data)
        print(f"Total number of logged entries: {len(self.data)}")
        
        # Analyze first entry structure
        if self.data:
            first_entry = self.data[0]
            print(f"\nFirst entry has {len(first_entry)} fields:")
            
            self.analysis_results['fields'] = {}
            for key, value in first_entry.items():
                type_info = self.get_data_type_info(value)
                self.analysis_results['fields'][key] = type_info
                print(f"  '{key}': {type_info}")
        
        # Check data consistency across entries
        self._check_data_consistency()
    
    def _check_data_consistency(self):
        """Check if all entries have the same structure."""
        if len(self.data) < 2:
            return
        
        print(f"\n📊 CONSISTENCY CHECK (across {len(self.data)} entries):")
        
        first_keys = set(self.data[0].keys())
        inconsistent_entries = []
        
        for i, entry in enumerate(self.data[1:], 1):
            entry_keys = set(entry.keys())
            if entry_keys != first_keys:
                inconsistent_entries.append(i)
        
        if inconsistent_entries:
            print(f"⚠️  Found {len(inconsistent_entries)} entries with different structure")
            print(f"   Inconsistent entries: {inconsistent_entries[:10]}{'...' if len(inconsistent_entries) > 10 else ''}")
        else:
            print("✅ All entries have consistent structure")
    
    def analyze_robot_state_data(self):
        """Analyze robot state data (joint positions, velocities, etc.)."""
        if not self.data:
            return
        
        print("\n" + "="*80)
        print("ROBOT STATE DATA ANALYSIS")
        print("="*80)
        
        robot_keys = ['qpos', 'qvel', 'qpos_des', 'gripper_qpos', 'gripper_qpos_des', 'gripper_target']
        found_robot_keys = [key for key in robot_keys if key in self.data[0]]
        
        if not found_robot_keys:
            print("❌ No robot state data found")
            return
        
        print(f"✅ Found robot state fields: {found_robot_keys}")
        
        for key in found_robot_keys:
            print(f"\n📈 Analyzing '{key}':")
            self._analyze_robot_field(key)
    
    def _analyze_robot_field(self, field_name: str):
        """Analyze a specific robot state field."""
        sample_data = self.data[0][field_name]
        
        if isinstance(sample_data, dict):
            print(f"   Structure: Dictionary with arms: {list(sample_data.keys())}")
            
            for arm_name, arm_data in sample_data.items():
                if isinstance(arm_data, np.ndarray):
                    print(f"   '{arm_name}': {self.get_data_type_info(arm_data)}")
                    
                    # Check data range across all entries
                    all_values = []
                    for entry in self.data[:min(100, len(self.data))]:  # Sample first 100 entries
                        if field_name in entry and arm_name in entry[field_name]:
                            if entry[field_name][arm_name] is not None:
                                all_values.append(entry[field_name][arm_name])
                    
                    if all_values:
                        all_values = np.array(all_values)
                        print(f"      Range (first 100 entries): min={all_values.min():.4f}, max={all_values.max():.4f}")
                else:
                    print(f"   '{arm_name}': {self.get_data_type_info(arm_data)}")
        else:
            print(f"   Structure: {self.get_data_type_info(sample_data)}")
    
    def analyze_camera_data(self):
        """Analyze camera image data."""
        if not self.data:
            return
        
        print("\n" + "="*80)
        print("CAMERA DATA ANALYSIS")
        print("="*80)
        
        image_keys = ['image', 'images', 'camera_frames', 'frames']
        found_image_key = None
        
        for key in image_keys:
            if key in self.data[0]:
                found_image_key = key
                break
        
        if not found_image_key:
            print("❌ No camera/image data found")
            return
        
        print(f"✅ Found camera data field: '{found_image_key}'")
        
        sample_images = self.data[0][found_image_key]
        
        if isinstance(sample_images, dict):
            print(f"   Camera structure: Dictionary with cameras: {list(sample_images.keys())}")
            
            for camera_name, image_data in sample_images.items():
                print(f"   '{camera_name}': {self.get_data_type_info(image_data)}")
                
                if isinstance(image_data, np.ndarray) and len(image_data.shape) >= 2:
                    print(f"      Image dimensions: {image_data.shape}")
                    print(f"      Data type: {image_data.dtype}")
                    if len(image_data.shape) == 3:
                        print(f"      Color channels: {image_data.shape[2]}")
        else:
            print(f"   Structure: {self.get_data_type_info(sample_images)}")
    
    def analyze_timestamps(self):
        """Analyze timing information."""
        if not self.data:
            return
        
        print("\n" + "="*80)
        print("TIMESTAMP ANALYSIS")
        print("="*80)
        
        if 'timestamp' not in self.data[0]:
            print("❌ No timestamp field found")
            return
        
        print("✅ Found timestamp data")
        
        timestamps = [entry['timestamp'] for entry in self.data if 'timestamp' in entry]
        
        if len(timestamps) > 1:
            timestamps = np.array(timestamps)
            duration = timestamps[-1] - timestamps[0]
            avg_freq = len(timestamps) / duration if duration > 0 else 0
            
            print(f"   Recording duration: {duration:.2f} seconds")
            print(f"   Average frequency: {avg_freq:.1f} Hz")
            print(f"   First timestamp: {timestamps[0]:.3f}")
            print(f"   Last timestamp: {timestamps[-1]:.3f}")
            
            # Analyze frequency consistency
            if len(timestamps) > 2:
                dt_values = np.diff(timestamps)
                print(f"   Time step statistics:")
                print(f"      Mean dt: {dt_values.mean():.4f}s ({1/dt_values.mean():.1f} Hz)")
                print(f"      Std dt: {dt_values.std():.4f}s")
                print(f"      Min dt: {dt_values.min():.4f}s")
                print(f"      Max dt: {dt_values.max():.4f}s")
    
    def show_sample_data(self, num_samples: int = 3):
        """Show sample data from a few entries."""
        if not self.data:
            return
        
        print("\n" + "="*80)
        print(f"SAMPLE DATA (first {min(num_samples, len(self.data))} entries)")
        print("="*80)
        
        for i in range(min(num_samples, len(self.data))):
            print(f"\n📋 Entry {i}:")
            entry = self.data[i]
            
            for key, value in entry.items():
                if key == 'timestamp':
                    print(f"   {key}: {value:.6f}")
                elif key == 'image' and isinstance(value, dict):
                    print(f"   {key}: dict with {len(value)} cameras")
                    for cam_name in value.keys():
                        if isinstance(value[cam_name], np.ndarray):
                            print(f"      {cam_name}: array{value[cam_name].shape}")
                elif isinstance(value, dict):
                    print(f"   {key}: dict({list(value.keys())})")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, np.ndarray):
                            print(f"      {sub_key}: {self.get_data_type_info(sub_value)}")
                        elif sub_value is None:
                            print(f"      {sub_key}: None")
                        else:
                            print(f"      {sub_key}: {type(sub_value).__name__}")
                elif isinstance(value, np.ndarray):
                    print(f"   {key}: {self.get_data_type_info(value)}")
                else:
                    print(f"   {key}: {type(value).__name__}")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        if not self.data:
            return
        
        print("\n" + "="*80)
        print("COMPREHENSIVE SUMMARY REPORT")
        print("="*80)
        
        print(f"📄 File: {self.pkl_file_path}")
        print(f"📊 Total entries: {len(self.data)}")
        print(f"💾 File size: {self.pkl_file_path.stat().st_size / (1024*1024):.2f} MB")
        
        if 'timestamp' in self.data[0] and len(self.data) > 1:
            duration = self.data[-1]['timestamp'] - self.data[0]['timestamp']
            print(f"⏱️  Duration: {duration:.2f} seconds")
            print(f"📈 Average logging frequency: {len(self.data) / duration:.1f} Hz")
        
        print(f"\n🔧 Data fields found:")
        for field, type_info in self.analysis_results.get('fields', {}).items():
            print(f"   • {field}: {type_info}")
        
        # Check for key teleoperation data types
        key_data_types = {
            'Robot Joint Data': ['qpos', 'qvel', 'qpos_des'],
            'Gripper Data': ['gripper_qpos', 'gripper_target', 'gripper_qpos_des'],
            'Camera Data': ['image', 'images', 'frames'],
            'Timing Data': ['timestamp']
        }
        
        print(f"\n📋 Data type summary:")
        for category, keys in key_data_types.items():
            found_keys = [key for key in keys if key in self.data[0]]
            if found_keys:
                print(f"   ✅ {category}: {found_keys}")
            else:
                print(f"   ❌ {category}: Not found")
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("🤖 XRoboToolkit Teleop Data Log Analyzer")
        print("="*80)
        
        if not self.load_data():
            return False
        
        self.analyze_data_structure()
        self.analyze_robot_state_data()
        self.analyze_camera_data()
        self.analyze_timestamps()
        self.show_sample_data()
        self.generate_summary_report()
        
        return True


def find_first_pkl_file(logs_dir: str = "logs") -> Optional[str]:
    """Find the first .pkl file in the logs directory."""
    logs_path = Path(logs_dir)
    
    if not logs_path.exists():
        print(f"❌ Logs directory '{logs_dir}' not found")
        return None
    
    # Search for .pkl files recursively
    pkl_files = list(logs_path.rglob("*.pkl"))
    
    if not pkl_files:
        print(f"❌ No .pkl files found in '{logs_dir}'")
        return None
    
    return str(pkl_files[0])


def main():
    """Main function to run the analysis."""
    # Determine which file to analyze
    if len(sys.argv) > 1:
        pkl_file_path = sys.argv[1]
    else:
        pkl_file_path = find_first_pkl_file()
        if not pkl_file_path:
            return 1
    
    # Check if file exists
    if not Path(pkl_file_path).exists():
        print(f"❌ File not found: {pkl_file_path}")
        return 1
    
    # Run analysis
    analyzer = TeleopDataAnalyzer(pkl_file_path)
    success = analyzer.run_full_analysis()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)